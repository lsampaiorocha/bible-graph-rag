import { aliasKey } from "./aliases.ts";
import {
	type Embedder,
	type GraphSource,
	KIN_TYPES,
	type LinkRow,
	MAX_LINEAGE_DEPTH,
	RELATIONS,
	type Relation,
} from "./graphSource.ts";
import {
	DEFAULT_LOCAL_PARAMS,
	expandFrom,
	type SeedMatch,
	type Subgraph,
} from "./modes/local.ts";
import {
	FAMILY_PATH_MAX_LINKS,
	PATH_MAX_LINKS,
	shortestPaths,
} from "./paths.ts";
import { rankSubgraph } from "./rank.ts";
import type { TraceStep } from "./trace.ts";

export const TOOLS = [
	"find_entity",
	"explore",
	"ancestors",
	"descendants",
	"relatives",
	"path",
	"overview",
] as const;
export type ToolName = (typeof TOOLS)[number];
export const MAX_TOOL_CALLS = 3;
// On the 0..1 scale of cosineScore. Measured on Genesis: misspelled names and plain
// questions land at 0.835-0.89, an unrelated phrase at 0.794.
export const MEANING_MATCH_MIN = 0.82;

export interface ToolContext {
	embed?: Embedder;
	questionVector?: number[] | null;
}

export interface ToolCall {
	tool: ToolName;
	entity: string;
	relation?: Relation | null;
	generations?: number | null;
	other?: string | null;
	family_only?: boolean | null;
}

const ENTITY_ID = /^(per|pla|grp|evt)_[a-z0-9_]+$/;
// A book's name with no entity of that name (Genesis, unlike Job or Ruth) is not
// matched by meaning, where it lands on someone ("Genesis" comes out as Noah).
const BOOK_NAME =
	/^(the )?(book of )?(bible|genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|(1|2|first|second) (samuel|kings|chronicles)|ezra|nehemiah|esther|job|psalms?|proverbs|ecclesiastes|song of (songs|solomon)|isaiah|jeremiah|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|(1|2|first|second) (corinthians|thessalonians|timothy|peter)|(1|2|3) john|galatians|ephesians|philippians|colossians|titus|philemon|hebrews|james|jude|revelation)$/i;
const PERSON_TOOLS = new Set<ToolName>([
	"ancestors",
	"descendants",
	"relatives",
]);

interface Resolution {
	id?: string;
	label?: string;
	candidates: string[];
	note?: string;
}

async function resolveEntity(
	source: GraphSource,
	call: ToolCall,
	context: ToolContext,
): Promise<Resolution> {
	if (ENTITY_ID.test(call.entity)) {
		const [card] = await source.entityCards([call.entity]);
		return card
			? { id: card.id, label: card.label, candidates: [] }
			: { candidates: [], note: `no entity with id ${call.entity}` };
	}
	const hits = (await source.seedsByAlias([aliasKey(call.entity)])).filter(
		(hit) => !PERSON_TOOLS.has(call.tool) || hit.nodeId.startsWith("per_"),
	);
	const [best] = hits;
	if (!best) {
		if (BOOK_NAME.test(call.entity.trim()))
			return {
				candidates: [],
				note: `${call.entity} is a book of the Bible, not a person, place or event`,
			};
		// No name matched (a misspelling, a description): fall back to the nearest entity
		// by meaning, if it is close enough to trust.
		const nearest = context.embed
			? (await source.seedsByVector(await context.embed(call.entity), 5)).find(
					(hit) =>
						!hit.nodeId.startsWith("evt_") &&
						(!PERSON_TOOLS.has(call.tool) || hit.nodeId.startsWith("per_")),
				)
			: undefined;
		if (nearest && nearest.score >= MEANING_MATCH_MIN) {
			return {
				id: nearest.nodeId,
				label: nearest.label,
				candidates: [],
				note: `no entity named "${call.entity}"; matched by meaning to ${nearest.label}`,
			};
		}
		return { candidates: [], note: `no entity named "${call.entity}"` };
	}
	// Several people can share a name; take the most mentioned and record the rest so a
	// second planning round (or the reader) can pick another.
	const candidates = hits.length > 1 ? hits.map((hit) => hit.nodeId) : [];
	return {
		id: best.nodeId,
		label: best.label,
		candidates,
		note: candidates.length
			? `${hits.length} entities are named "${call.entity}"`
			: undefined,
	};
}

function addLinks(
	subgraph: Subgraph,
	rows: LinkRow[],
	root: { id: string; label: string },
): void {
	const labels = new Map([[root.id, root.label]]);
	const adjacency = new Map<string, { edgeId: string; neighbour: string }[]>();
	for (const row of rows) {
		labels.set(row.source, row.sourceLabel);
		labels.set(row.target, row.targetLabel);
		for (const [from, to] of [
			[row.source, row.target],
			[row.target, row.source],
		] as const) {
			adjacency.set(from, [
				...(adjacency.get(from) ?? []),
				{ edgeId: row.edgeId, neighbour: to },
			]);
		}
	}
	// Breadth-first distance from the root over the returned links: the generation for a
	// lineage, and 2 for a sibling found through a shared parent.
	const hops = new Map([[root.id, 0]]);
	const via = new Map<string, string>();
	const queue = [root.id];
	for (let id = queue.shift(); id !== undefined; id = queue.shift()) {
		for (const { edgeId, neighbour } of adjacency.get(id) ?? []) {
			if (!hops.has(neighbour)) {
				hops.set(neighbour, (hops.get(id) ?? 0) + 1);
				via.set(neighbour, edgeId);
				queue.push(neighbour);
			}
		}
	}
	for (const [id, distance] of hops) {
		const existing = subgraph.nodes.get(id);
		const hop = existing ? Math.min(existing.hop, distance) : distance;
		const reachedVia = existing?.reached_via ?? via.get(id);
		subgraph.nodes.set(id, {
			id,
			label: labels.get(id) ?? id,
			hop,
			score: 1 / (1 + hop),
			status: "context",
			...(reachedVia ? { reached_via: reachedVia } : {}),
		});
	}
	for (const row of rows) {
		if (!subgraph.edges.has(row.edgeId)) {
			subgraph.edges.set(row.edgeId, {
				id: row.edgeId,
				source: row.source,
				target: row.target,
				type: row.type,
				hop: Math.max(hops.get(row.source) ?? 0, hops.get(row.target) ?? 0),
				score: row.supportCount,
				status: "context",
				evidence_chunk_ids: [],
			});
		}
	}
}

export async function executeCall(
	source: GraphSource,
	call: ToolCall,
	index: number,
	into: Subgraph,
	context: ToolContext = {},
): Promise<TraceStep> {
	const started = performance.now();
	if (call.tool === "overview")
		return overviewStep(source, call, index, into, started);
	const args: Record<string, string | number> = { entity: call.entity };
	if (call.relation) args.relation = call.relation;
	if (call.generations) args.generations = call.generations;
	if (call.other) args.other = call.other;
	if (call.family_only) args.family_only = "yes";
	const resolution = await resolveEntity(source, call, context);
	const touched = { nodes: new Set<string>(), edges: new Set<string>() };
	let pathNote: string | undefined;

	if (resolution.id && resolution.label) {
		const root = { id: resolution.id, label: resolution.label };
		if (call.tool === "find_entity") {
			const cards = await source.entityCards([
				root.id,
				...resolution.candidates,
			]);
			for (const card of cards) {
				if (!into.nodes.has(card.id)) {
					into.nodes.set(card.id, {
						id: card.id,
						label: card.label,
						hop: 0,
						score: 1,
						status: "context",
					});
				}
				touched.nodes.add(card.id);
			}
		} else if (call.tool === "explore") {
			const local = await expandFrom(source, [root], DEFAULT_LOCAL_PARAMS);
			await rankSubgraph(
				source,
				local,
				[root.id],
				context.questionVector ?? null,
			);
			for (const [id, node] of local.nodes) {
				if (
					!into.nodes.has(id) ||
					(node.status === "context" &&
						into.nodes.get(id)?.status === "candidate")
				) {
					into.nodes.set(id, node);
				}
				touched.nodes.add(id);
			}
			for (const [id, edge] of local.edges) {
				if (
					!into.edges.has(id) ||
					(edge.status === "context" &&
						into.edges.get(id)?.status === "candidate")
				) {
					into.edges.set(id, edge);
				}
				touched.edges.add(id);
			}
		} else if (call.tool === "path") {
			const other: Resolution = call.other
				? await resolveEntity(source, { ...call, entity: call.other }, context)
				: { candidates: [], note: "path needs other, the second entity" };
			if (other.id) {
				const maxLinks = call.family_only
					? FAMILY_PATH_MAX_LINKS
					: PATH_MAX_LINKS;
				const rows = await source.corridor(
					[root.id, other.id],
					Math.ceil(maxLinks / 2),
					call.family_only ? KIN_TYPES : null,
				);
				const found = shortestPaths(rows, root.id, other.id, maxLinks);
				if (found.length === 0) pathNote = `no path within ${maxLinks} links`;
				const used = new Set(found.flatMap((path) => path.edgeIds));
				const pathRows = rows.filter((row) => used.has(row.edgeId));
				addLinks(into, pathRows, root);
				const ranked = into.paths?.length ?? 0;
				into.paths = [
					...(into.paths ?? []),
					...found.map((path, rank) => ({
						rank: ranked + rank + 1,
						cost: path.cost,
						status: "context" as const,
						node_ids: path.nodeIds,
						edge_ids: path.edgeIds,
					})),
				];
				for (const row of pathRows) {
					touched.nodes.add(row.source);
					touched.nodes.add(row.target);
					touched.edges.add(row.edgeId);
				}
			} else {
				pathNote = other.note;
			}
		} else {
			const rows =
				call.tool === "relatives"
					? await source.relatives(root.id, call.relation ?? "children")
					: await source.lineage(
							root.id,
							call.tool,
							Math.min(
								call.generations ?? MAX_LINEAGE_DEPTH,
								MAX_LINEAGE_DEPTH,
							),
						);
			addLinks(into, rows, root);
			touched.nodes.add(root.id);
			for (const row of rows) {
				touched.nodes.add(row.source);
				touched.nodes.add(row.target);
				touched.edges.add(row.edgeId);
			}
		}
	}

	const step: TraceStep = {
		index,
		tool: call.tool,
		args,
		node_ids: [...touched.nodes].sort(),
		edge_ids: [...touched.edges].sort(),
		ms: Math.round(performance.now() - started),
	};
	if (resolution.id) step.entity_id = resolution.id;
	if (resolution.candidates.length) step.candidates = resolution.candidates;
	const note =
		resolution.note ??
		pathNote ??
		(touched.edges.size === 0 && call.tool !== "find_entity"
			? "no links found"
			: undefined);
	if (note) step.note = note;
	return step;
}

const CONNECTION_WORDS =
	/\b(related|relation|relationship|connected|connection|kin)\b/;
const KINSHIP_WORDS = /\b(related|relation|relationship|kin)\b/;
async function overviewStep(
	source: GraphSource,
	call: ToolCall,
	index: number,
	into: Subgraph,
	started: number,
): Promise<TraceStep> {
	const kind = /event|stor|happen/i.test(call.entity) ? "event" : "person";
	const cards = await source.overview(kind);
	const ids = cards.map((card) => card.id);
	// Main characters come with the links among them. Main events come with the main
	// characters who took part, which keeps the view to a readable story map.
	const mainCharacters =
		kind === "event"
			? new Set((await source.overview("person")).map((card) => card.id))
			: null;
	const rows =
		kind === "person"
			? await source.corridor(ids, 0, null)
			: (await source.corridor(ids, 1, ["PARTICIPATED_IN"])).filter((row) =>
					mainCharacters?.has(row.source),
				);
	cards.forEach((card, rank) => {
		into.nodes.set(card.id, {
			id: card.id,
			label: card.label,
			hop: 0,
			score: 1 / (1 + rank),
			status: "context",
		});
	});
	for (const row of rows) {
		for (const [id, label] of [
			[row.source, row.sourceLabel],
			[row.target, row.targetLabel],
		] as const) {
			if (!into.nodes.has(id))
				into.nodes.set(id, {
					id,
					label,
					hop: 1,
					score: 0.5,
					status: "context",
				});
		}
		if (!into.edges.has(row.edgeId))
			into.edges.set(row.edgeId, {
				id: row.edgeId,
				source: row.source,
				target: row.target,
				type: row.type,
				hop: 1,
				score: row.supportCount,
				status: "context",
				evidence_chunk_ids: [],
			});
	}
	const step: TraceStep = {
		index,
		tool: "overview",
		args: { entity: kind === "event" ? "events" : "people" },
		node_ids: [
			...new Set([...ids, ...rows.flatMap((row) => [row.source, row.target])]),
		].sort(),
		edge_ids: rows.map((row) => row.edgeId).sort(),
		ms: Math.round(performance.now() - started),
	};
	if (cards.length === 0) step.note = "no overview ranks in this graph";
	return step;
}

export const OVERVIEW_WORDS =
	/\b(main|most important|key|major|principal|central)\b.*\b(characters?|people|persons|figures|events?|stories|story)\b/;
const ANCESTOR_WORDS =
	/\b(ancestors?|ancestry|forefathers?|fathers? of .+ fathers?|lineage|genealogy|family line|descended from|line of)\b/;
const DESCENDANT_WORDS =
	/\b(descendants?|offspring|posterity|generations of)\b/;
const RELATION_WORDS: [RegExp, Relation][] = [
	[/\b(wife|wives|husband|husbands|spouses?|marry|married)\b/, "spouses"],
	[/\b(sons?)\b/, "sons"],
	[/\b(daughters?)\b/, "daughters"],
	[/\b(children|child|kids)\b/, "children"],
	[/\b(brothers?)\b/, "brothers"],
	[/\b(sisters?)\b/, "sisters"],
	[/\b(siblings?)\b/, "siblings"],
	[/\b(father)\b/, "father"],
	[/\b(mother)\b/, "mother"],
	[/\b(parents?)\b/, "parents"],
];

// Used without an LLM (the public site without a key) and when the planner fails:
// maps family words to the same tools the agent would call.
export function planByKeywords(
	question: string,
	seeds: SeedMatch[],
): ToolCall[] {
	const text = question.toLowerCase();
	const overview = OVERVIEW_WORDS.exec(text);
	if (overview)
		return [
			{
				tool: "overview",
				entity: /event|stor/.test(overview[2] ?? "") ? "events" : "people",
			},
		];
	const people = seeds.filter(({ seed }) => seed.node_id.startsWith("per_"));
	const subject = people[0] ?? seeds[0];
	if (!subject) return [];
	const entity = subject.seed.node_id;
	const [first, second] = seeds;
	if (first && second && CONNECTION_WORDS.test(text))
		return [
			{
				tool: "path",
				entity: first.seed.node_id,
				other: second.seed.node_id,
				family_only: KINSHIP_WORDS.test(text),
			},
		];
	if (people.length && ANCESTOR_WORDS.test(text))
		return [{ tool: "ancestors", entity }];
	if (people.length && DESCENDANT_WORDS.test(text))
		return [{ tool: "descendants", entity }];
	const relation = RELATION_WORDS.find(([pattern]) => pattern.test(text))?.[1];
	if (people.length && relation && RELATIONS.includes(relation))
		return [{ tool: "relatives", entity, relation }];
	return seeds
		.slice(0, MAX_TOOL_CALLS)
		.map(({ seed }) => ({ tool: "explore" as const, entity: seed.node_id }));
}
