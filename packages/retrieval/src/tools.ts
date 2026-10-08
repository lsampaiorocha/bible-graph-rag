import { aliasKey } from "./aliases.ts";
import {
	type GraphSource,
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
import type { TraceStep } from "./trace.ts";

export const TOOLS = [
	"find_entity",
	"explore",
	"ancestors",
	"descendants",
	"relatives",
] as const;
export type ToolName = (typeof TOOLS)[number];
export const MAX_TOOL_CALLS = 3;

export interface ToolCall {
	tool: ToolName;
	entity: string;
	relation?: Relation | null;
	generations?: number | null;
}

const ENTITY_ID = /^(per|pla|grp|evt)_[a-z0-9_]+$/;
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
	if (!best)
		return { candidates: [], note: `no entity named "${call.entity}"` };
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
): Promise<TraceStep> {
	const started = performance.now();
	const args: Record<string, string | number> = { entity: call.entity };
	if (call.relation) args.relation = call.relation;
	if (call.generations) args.generations = call.generations;
	const resolution = await resolveEntity(source, call);
	const touched = { nodes: new Set<string>(), edges: new Set<string>() };

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
		(touched.edges.size === 0 && call.tool !== "find_entity"
			? "no links found"
			: undefined);
	if (note) step.note = note;
	return step;
}

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
	const people = seeds.filter(({ seed }) => seed.node_id.startsWith("per_"));
	const subject = people[0] ?? seeds[0];
	if (!subject) return [];
	const entity = subject.seed.node_id;
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
