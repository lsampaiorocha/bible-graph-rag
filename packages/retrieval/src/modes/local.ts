import { questionNgrams } from "../aliases.ts";
import { compareText, type GraphSource } from "../graphSource.ts";
import type { RetrievalTrace, Seed, TraceEdge, TraceNode } from "../trace.ts";

export interface LocalParams {
	maxHops: number;
	topM: number;
}

export const DEFAULT_LOCAL_PARAMS: LocalParams = { maxHops: 2, topM: 8 };

export interface SeedMatch {
	seed: Seed;
	label: string;
}

export async function findSeeds(
	source: GraphSource,
	question: string,
): Promise<SeedMatch[]> {
	const ngrams = questionNgrams(question);
	const hits = await source.seedsByAlias([
		...new Set(ngrams.map((ngram) => ngram.key)),
	]);
	const hitsByKey = Map.groupBy(hits, (hit) => hit.key);
	// Longest match first, so "padan aram" wins over the person "Aram" inside it.
	const taken = new Set<number>();
	const seeds = new Map<string, SeedMatch>();
	for (const ngram of ngrams) {
		const keyHits = hitsByKey.get(ngram.key);
		const span = Array.from(
			{ length: ngram.end - ngram.start },
			(_, offset) => ngram.start + offset,
		);
		if (!keyHits || span.some((position) => taken.has(position))) continue;
		for (const position of span) taken.add(position);
		for (const hit of keyHits) {
			if (!seeds.has(hit.nodeId)) {
				seeds.set(hit.nodeId, {
					seed: {
						node_id: hit.nodeId,
						method: "alias_exact",
						matched_text: ngram.key,
						score: 1,
					},
					label: hit.label,
				});
			}
		}
	}
	return [...seeds.values()];
}

export interface Subgraph {
	nodes: Map<string, TraceNode>;
	edges: Map<string, TraceEdge>;
}

export async function expandFrom(
	source: GraphSource,
	seeds: { id: string; label: string }[],
	params: LocalParams,
): Promise<Subgraph> {
	const nodes = new Map<string, TraceNode>();
	const edges = new Map<string, TraceEdge>();
	for (const { id, label } of seeds) {
		nodes.set(id, { id, label, hop: 0, score: 1, status: "context" });
	}

	let frontier = [...nodes.keys()];
	for (let hop = 1; hop <= params.maxHops && frontier.length > 0; hop++) {
		const rows = await source.expand(frontier, 2 * params.topM);
		const nextFrontier: string[] = [];
		for (const fromRows of Map.groupBy(rows, (row) => row.fromId).values()) {
			fromRows.forEach((row, rank) => {
				const kept = rank < params.topM;
				const status = kept ? "context" : "candidate";
				const pruned = kept ? {} : { pruned_reason: "neighbour_cap" as const };
				const existingEdge = edges.get(row.edgeId);
				if (!existingEdge || (kept && existingEdge.status === "candidate")) {
					edges.set(row.edgeId, {
						id: row.edgeId,
						source: row.source,
						target: row.target,
						type: row.type,
						hop,
						score: row.supportCount,
						status,
						evidence_chunk_ids: [],
						...pruned,
					});
				}
				const existingNode = nodes.get(row.neighbourId);
				if (!existingNode) {
					nodes.set(row.neighbourId, {
						id: row.neighbourId,
						label: row.neighbourLabel,
						hop,
						score: 1 / (1 + hop),
						status,
						reached_via: row.edgeId,
						...pruned,
					});
					if (kept) nextFrontier.push(row.neighbourId);
				} else if (kept && existingNode.status === "candidate") {
					const { pruned_reason: _dropped, ...promoted } = existingNode;
					nodes.set(row.neighbourId, { ...promoted, status: "context" });
					nextFrontier.push(row.neighbourId);
				}
			});
		}
		frontier = nextFrontier;
	}
	return { nodes, edges };
}

export function orderSubgraph(subgraph: Subgraph): {
	nodes: TraceNode[];
	edges: TraceEdge[];
	handles: Record<string, string>;
} {
	const maxSupport = Math.max(
		1,
		...[...subgraph.edges.values()].map((edge) => edge.score),
	);
	const nodes = [...subgraph.nodes.values()].sort(
		(a, b) => a.hop - b.hop || compareText(a.id, b.id),
	);
	const edges = [...subgraph.edges.values()]
		.map((edge) => ({ ...edge, score: edge.score / maxSupport }))
		.sort((a, b) => a.hop - b.hop || compareText(a.id, b.id));
	const handles: Record<string, string> = {};
	for (const [prefix, elements] of [
		["E", nodes],
		["R", edges],
	] as const) {
		let next = 1;
		for (const element of elements) {
			if (element.status !== "candidate") {
				element.handle = `${prefix}${next++}`;
				handles[element.handle] = element.id;
			}
		}
	}
	return { nodes, edges, handles };
}

export async function runLocal(
	source: GraphSource,
	question: string,
	params: LocalParams = DEFAULT_LOCAL_PARAMS,
): Promise<RetrievalTrace> {
	const started = performance.now();
	const seedMatches = await findSeeds(source, question);
	const seeded = performance.now();
	const subgraph = await expandFrom(
		source,
		seedMatches.map(({ seed, label }) => ({ id: seed.node_id, label })),
		params,
	);
	const expanded = performance.now();
	const { nodes, edges, handles } = orderSubgraph(subgraph);

	return {
		trace_version: "1.0",
		trace_id: crypto.randomUUID(),
		index_id: await source.indexId(),
		created_at: new Date().toISOString(),
		source: "live",
		query: {
			text: question,
			mode: "local",
			params: { max_hops: params.maxHops, top_m: params.topM },
		},
		seeds: seedMatches.map(({ seed }) => seed),
		nodes,
		edges,
		paths: [],
		communities: [],
		chunks: [],
		context: { handles, token_count: 0, token_budget: 0, truncated: false },
		answer: null,
		metrics: {
			timings_ms: {
				seed: Math.round(seeded - started),
				expand: Math.round(expanded - seeded),
			},
		},
	};
}
