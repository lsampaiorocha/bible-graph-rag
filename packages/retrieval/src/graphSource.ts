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
}

// Two implementations exist: Neo4j for the local API and in-memory JSON for the
// browser. Every ordering breaks ties by id so both return identical rows.
export interface GraphSource {
	indexId(): Promise<string>;
	// Hits ordered by key, then mentionCount descending, then nodeId.
	seedsByAlias(keys: string[]): Promise<AliasHit[]>;
	// Per frontier node, its neighbours ordered by supportCount descending then edgeId,
	// at most fetchPerNode of them; rows ordered by fromId first.
	expand(frontierIds: string[], fetchPerNode: number): Promise<ExpansionRow[]>;
}

export interface MemoryGraph {
	indexId: string;
	entities: { id: string; name: string; mentionCount: number }[];
	edges: {
		id: string;
		type: string;
		source: string;
		target: string;
		supportCount: number;
	}[];
	aliases: { key: string; nodeId: string }[];
}

export class MemoryGraphSource implements GraphSource {
	private readonly graph: MemoryGraph;
	private readonly entityById: Map<string, MemoryGraph["entities"][number]>;

	constructor(graph: MemoryGraph) {
		this.graph = graph;
		this.entityById = new Map(
			graph.entities.map((entity) => [entity.id, entity]),
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
					};
				}),
		);
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
