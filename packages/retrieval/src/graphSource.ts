export interface AliasHit {
	key: string;
	nodeId: string;
	label: string;
	mentionCount: number;
}

export interface ExpansionRow {
	fromId: string;
	edgeId: string;
	type: string;
	source: string;
	target: string;
	supportCount: number;
	neighbourId: string;
	neighbourLabel: string;
	neighbourDegree: number;
}

export interface VectorHit {
	nodeId: string;
	label: string;
	score: number;
}

export interface PassageHit {
	id: string;
	text: string;
	score: number;
	verseRefs: string[];
}

export type Embedder = (text: string) => Promise<number[]>;

// Neo4j reports cosine similarity rescaled to 0..1 as (1 + cos) / 2; the in-memory source
// uses the same scale so both rank and threshold identically.
export function cosineScore(a: number[], b: number[]): number {
	let dot = 0;
	let normA = 0;
	let normB = 0;
	a.forEach((value, index) => {
		const other = b[index] ?? 0;
		dot += value * other;
		normA += value * value;
		normB += other * other;
	});
	return normA && normB ? (1 + dot / Math.sqrt(normA * normB)) / 2 : 0;
}

export interface EntityCard {
	id: string;
	label: string;
	kind: string;
	gender: string | null;
	description: string;
	firstVerse: string | null;
	mentionCount: number;
}

export interface EdgeDetail {
	id: string;
	description: string;
	evidenceVerses: string[];
	otherLabel: string | null;
}

// One family link found by a lineage or relatives query; depth = generations (or
// steps) between the asked-about person and the far end of the link.
export interface LinkRow {
	edgeId: string;
	type: string;
	source: string;
	target: string;
	sourceLabel: string;
	targetLabel: string;
	supportCount: number;
	depth: number;
}

export const KIN_TYPES = [
	"FATHER_OF",
	"MOTHER_OF",
	"SIBLING_OF",
	"SPOUSE_OF",
] as const;

export type LineageDirection = "ancestors" | "descendants";
export const RELATIONS = [
	"father",
	"mother",
	"parents",
	"children",
	"sons",
	"daughters",
	"spouses",
	"siblings",
	"brothers",
	"sisters",
] as const;
export type Relation = (typeof RELATIONS)[number];
// Genesis runs about 24 generations from Adam to Jacob's grandsons.
export const MAX_LINEAGE_DEPTH = 30;
export const MAX_LINEAGE_LINKS = 200;

// Two implementations exist: Neo4j for the local API and in-memory JSON for the
// browser. Every ordering breaks ties by id so both return identical rows.
export interface GraphSource {
	indexId(): Promise<string>;
	// Hits ordered by key, then mentionCount descending, then nodeId.
	seedsByAlias(keys: string[]): Promise<AliasHit[]>;
	// Per frontier node, its neighbours ordered by supportCount descending then edgeId,
	// at most fetchPerNode of them; rows ordered by fromId first.
	expand(frontierIds: string[], fetchPerNode: number): Promise<ExpansionRow[]>;
	// Parent links walked up or down from a person, ordered by depth then edgeId.
	lineage(
		personId: string,
		direction: LineageDirection,
		maxDepth: number,
	): Promise<LinkRow[]>;
	// Direct family links of one kind; siblings also come through a shared parent,
	// returned as that parent's two links (depth 1 to the person, 2 to the sibling).
	relatives(personId: string, relation: Relation): Promise<LinkRow[]>;
	// The book's main characters ("person") or the leading event of each story cluster
	// ("event"), in rank order, as ranked when the graph was built.
	overview(kind: "person" | "event"): Promise<EntityCard[]>;
	// Links among the entities within `reach` links of any given id, optionally only of
	// the given types: the corridor a path between those ids can run through.
	corridor(
		ids: string[],
		reach: number,
		types: readonly string[] | null,
	): Promise<LinkRow[]>;
	// Nearest entities to a vector, ordered by score descending then nodeId.
	seedsByVector(vector: number[], limit: number): Promise<VectorHit[]>;
	similarities(
		ids: string[],
		vector: number[],
	): Promise<Record<string, number>>;
	// Nearest passages to a vector, for the plain vector-RAG baseline.
	passagesByVector(vector: number[], limit: number): Promise<PassageHit[]>;
	entityCards(ids: string[]): Promise<EntityCard[]>;
	edgeDetails(ids: string[]): Promise<EdgeDetail[]>;
	verses(refs: string[]): Promise<{ id: string; text: string }[]>;
}

export interface MemoryGraph {
	indexId: string;
	entities: {
		id: string;
		name: string;
		mentionCount: number;
		kind?: string;
		gender?: string | null;
		description?: string;
		firstVerse?: string | null;
		embedding?: number[];
		overviewRank?: number | null;
	}[];
	edges: {
		id: string;
		type: string;
		source: string;
		target: string;
		supportCount: number;
		description?: string;
		evidenceVerses?: string[];
		otherLabel?: string | null;
	}[];
	aliases: { key: string; nodeId: string }[];
	verses?: Record<string, string>;
	passages?: {
		id: string;
		text: string;
		verseRefs: string[];
		embedding: number[];
	}[];
}

const PARENT_TYPES = new Set(["FATHER_OF", "MOTHER_OF"]);

export class MemoryGraphSource implements GraphSource {
	private readonly graph: MemoryGraph;
	private readonly entityById: Map<string, MemoryGraph["entities"][number]>;
	private readonly degreeById: Map<string, number>;

	constructor(graph: MemoryGraph) {
		this.graph = graph;
		this.entityById = new Map(
			graph.entities.map((entity) => [entity.id, entity]),
		);
		const neighbours = new Map<string, Set<string>>();
		for (const edge of graph.edges) {
			for (const [from, to] of [
				[edge.source, edge.target],
				[edge.target, edge.source],
			] as const) {
				neighbours.set(from, (neighbours.get(from) ?? new Set()).add(to));
			}
		}
		this.degreeById = new Map(
			[...neighbours].map(([id, set]) => [id, set.size]),
		);
	}

	async indexId(): Promise<string> {
		return this.graph.indexId;
	}

	async seedsByAlias(keys: string[]): Promise<AliasHit[]> {
		const wanted = new Set(keys);
		return this.graph.aliases
			.filter((alias) => wanted.has(alias.key))
			.map((alias) => {
				const entity = this.entityOrThrow(alias.nodeId);
				return {
					key: alias.key,
					nodeId: entity.id,
					label: entity.name,
					mentionCount: entity.mentionCount,
				};
			})
			.sort(
				(a, b) =>
					compareText(a.key, b.key) ||
					b.mentionCount - a.mentionCount ||
					compareText(a.nodeId, b.nodeId),
			);
	}

	async expand(
		frontierIds: string[],
		fetchPerNode: number,
	): Promise<ExpansionRow[]> {
		return [...new Set(frontierIds)].sort(compareText).flatMap((fromId) =>
			this.graph.edges
				.filter((edge) => edge.source === fromId || edge.target === fromId)
				.sort(
					(a, b) => b.supportCount - a.supportCount || compareText(a.id, b.id),
				)
				.slice(0, fetchPerNode)
				.map((edge) => {
					const neighbour = this.entityOrThrow(
						edge.source === fromId ? edge.target : edge.source,
					);
					return {
						fromId,
						edgeId: edge.id,
						type: edge.type,
						source: edge.source,
						target: edge.target,
						supportCount: edge.supportCount,
						neighbourId: neighbour.id,
						neighbourLabel: neighbour.name,
						neighbourDegree: this.degreeById.get(neighbour.id) ?? 0,
					};
				}),
		);
	}

	async lineage(
		personId: string,
		direction: LineageDirection,
		maxDepth: number,
	): Promise<LinkRow[]> {
		const depthByEdge = new Map<string, number>();
		let frontier = [personId];
		for (
			let depth = 1;
			depth <= Math.min(maxDepth, MAX_LINEAGE_DEPTH) && frontier.length > 0;
			depth++
		) {
			const next: string[] = [];
			for (const edge of this.graph.edges) {
				if (!PARENT_TYPES.has(edge.type) || depthByEdge.has(edge.id)) continue;
				const near = direction === "ancestors" ? edge.target : edge.source;
				if (frontier.includes(near)) {
					depthByEdge.set(edge.id, depth);
					next.push(direction === "ancestors" ? edge.source : edge.target);
				}
			}
			frontier = next;
		}
		return this.linkRows(depthByEdge).slice(0, MAX_LINEAGE_LINKS);
	}

	async overview(kind: "person" | "event"): Promise<EntityCard[]> {
		const ranked = this.graph.entities
			.filter((entity) => entity.kind === kind && entity.overviewRank)
			.sort((a, b) => (a.overviewRank ?? 0) - (b.overviewRank ?? 0));
		const cards = new Map(
			(await this.entityCards(ranked.map((entity) => entity.id))).map(
				(card) => [card.id, card],
			),
		);
		return ranked.flatMap((entity) => cards.get(entity.id) ?? []);
	}

	async corridor(
		ids: string[],
		reach: number,
		types: readonly string[] | null,
	): Promise<LinkRow[]> {
		const allowed = this.graph.edges.filter(
			(edge) => !types || types.includes(edge.type),
		);
		const inside = new Set(ids);
		let frontier = [...inside];
		for (let step = 0; step < reach && frontier.length > 0; step++) {
			const next: string[] = [];
			for (const edge of allowed) {
				for (const [near, far] of [
					[edge.source, edge.target],
					[edge.target, edge.source],
				] as const) {
					if (frontier.includes(near) && !inside.has(far)) {
						inside.add(far);
						next.push(far);
					}
				}
			}
			frontier = next;
		}
		return this.linkRows(
			new Map(
				allowed
					.filter((edge) => inside.has(edge.source) && inside.has(edge.target))
					.map((edge) => [edge.id, 0]),
			),
		);
	}

	async relatives(personId: string, relation: Relation): Promise<LinkRow[]> {
		const depthByEdge = new Map<string, number>();
		const genderOf = (id: string) => this.entityById.get(id)?.gender ?? null;
		const wantsGender = (id: string) =>
			relation === "sons" || relation === "brothers"
				? genderOf(id) === "male"
				: relation === "daughters" || relation === "sisters"
					? genderOf(id) === "female"
					: true;
		for (const edge of this.graph.edges) {
			const isParent = PARENT_TYPES.has(edge.type);
			if (
				(relation === "father" &&
					edge.type === "FATHER_OF" &&
					edge.target === personId) ||
				(relation === "mother" &&
					edge.type === "MOTHER_OF" &&
					edge.target === personId) ||
				(relation === "parents" && isParent && edge.target === personId) ||
				(["children", "sons", "daughters"].includes(relation) &&
					isParent &&
					edge.source === personId &&
					wantsGender(edge.target)) ||
				(relation === "spouses" &&
					edge.type === "SPOUSE_OF" &&
					[edge.source, edge.target].includes(personId))
			) {
				depthByEdge.set(edge.id, 1);
			}
		}
		if (["siblings", "brothers", "sisters"].includes(relation)) {
			for (const edge of this.graph.edges) {
				if (
					edge.type === "SIBLING_OF" &&
					[edge.source, edge.target].includes(personId)
				) {
					const other = edge.source === personId ? edge.target : edge.source;
					if (wantsGender(other)) depthByEdge.set(edge.id, 1);
				}
			}
			const parentLinks = this.graph.edges.filter(
				(edge) => PARENT_TYPES.has(edge.type) && edge.target === personId,
			);
			for (const parentLink of parentLinks) {
				for (const edge of this.graph.edges) {
					if (
						PARENT_TYPES.has(edge.type) &&
						edge.source === parentLink.source &&
						edge.target !== personId &&
						wantsGender(edge.target)
					) {
						depthByEdge.set(parentLink.id, 1);
						depthByEdge.set(edge.id, 2);
					}
				}
			}
		}
		return this.linkRows(depthByEdge);
	}

	async seedsByVector(vector: number[], limit: number): Promise<VectorHit[]> {
		return this.graph.entities
			.filter((entity) => entity.embedding)
			.map((entity) => ({
				nodeId: entity.id,
				label: entity.name,
				score: cosineScore(entity.embedding ?? [], vector),
			}))
			.sort((a, b) => b.score - a.score || compareText(a.nodeId, b.nodeId))
			.slice(0, limit);
	}

	async similarities(
		ids: string[],
		vector: number[],
	): Promise<Record<string, number>> {
		return Object.fromEntries(
			ids.flatMap((id) => {
				const embedding = this.entityById.get(id)?.embedding;
				return embedding ? [[id, cosineScore(embedding, vector)]] : [];
			}),
		);
	}

	async passagesByVector(
		vector: number[],
		limit: number,
	): Promise<PassageHit[]> {
		return (this.graph.passages ?? [])
			.map((passage) => ({
				id: passage.id,
				text: passage.text,
				score: cosineScore(passage.embedding, vector),
				verseRefs: passage.verseRefs,
			}))
			.sort((a, b) => b.score - a.score || compareText(a.id, b.id))
			.slice(0, limit);
	}

	async entityCards(ids: string[]): Promise<EntityCard[]> {
		return [...new Set(ids)]
			.sort(compareText)
			.flatMap((id) => this.entityById.get(id) ?? [])
			.map((entity) => ({
				id: entity.id,
				label: entity.name,
				kind: entity.kind ?? "person",
				gender: entity.gender ?? null,
				description: entity.description ?? "",
				firstVerse: entity.firstVerse ?? null,
				mentionCount: entity.mentionCount,
			}));
	}

	async edgeDetails(ids: string[]): Promise<EdgeDetail[]> {
		const wanted = new Set(ids);
		return this.graph.edges
			.filter((edge) => wanted.has(edge.id))
			.sort((a, b) => compareText(a.id, b.id))
			.map((edge) => ({
				id: edge.id,
				description: edge.description ?? "",
				evidenceVerses: edge.evidenceVerses ?? [],
				otherLabel: edge.otherLabel ?? null,
			}));
	}

	async verses(refs: string[]): Promise<{ id: string; text: string }[]> {
		return [...new Set(refs)].flatMap((id) => {
			const text = this.graph.verses?.[id];
			return text === undefined ? [] : [{ id, text }];
		});
	}

	private linkRows(depthByEdge: Map<string, number>): LinkRow[] {
		return this.graph.edges
			.filter((edge) => depthByEdge.has(edge.id))
			.map((edge) => ({
				edgeId: edge.id,
				type: edge.type,
				source: edge.source,
				target: edge.target,
				sourceLabel: this.entityOrThrow(edge.source).name,
				targetLabel: this.entityOrThrow(edge.target).name,
				supportCount: edge.supportCount,
				depth: depthByEdge.get(edge.id) ?? 0,
			}))
			.sort((a, b) => a.depth - b.depth || compareText(a.edgeId, b.edgeId));
	}

	private entityOrThrow(id: string) {
		const entity = this.entityById.get(id);
		if (!entity) throw new Error(`unknown entity ${id}`);
		return entity;
	}
}

// Plain code-unit comparison, the same order Neo4j uses for strings.
export function compareText(a: string, b: string): number {
	return a < b ? -1 : a > b ? 1 : 0;
}
