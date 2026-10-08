import { compareText, type GraphSource } from "./graphSource.ts";
import type { Subgraph } from "./modes/local.ts";
import { personalizedPageRank } from "./ppr.ts";

export const EXPLORE_KEEP = 25;
export const HUB_DAMPING = 0.5;
export const SIMILARITY_WEIGHT = 0.5;

function normalized(values: Map<string, number>): Map<string, number> {
	const all = [...values.values()];
	const low = Math.min(...all);
	const high = Math.max(...all);
	return new Map(
		[...values].map(([id, value]) => [
			id,
			high > low ? (value - low) / (high - low) : 1,
		]),
	);
}

// Ranks what an expansion reached and keeps the best `keep` entities. PageRank from the
// seeds measures closeness in the graph; dividing by degree^0.5 stops hubs such as God,
// linked to almost everyone, from winning every query; similarity to the question
// favours what the question is about.
export async function rankSubgraph(
	source: GraphSource,
	subgraph: Subgraph,
	seedIds: string[],
	questionVector: number[] | null,
	keep = EXPLORE_KEEP,
): Promise<void> {
	const seeds = new Set(seedIds);
	const edges = [...subgraph.edges.values()].map((edge) => ({
		id: edge.id,
		source: edge.source,
		target: edge.target,
		weight: Math.max(1, edge.score),
	}));
	const pagerank = personalizedPageRank(
		[...subgraph.nodes.keys()],
		edges,
		seedIds,
	);
	const others = [...subgraph.nodes.keys()].filter((id) => !seeds.has(id));
	if (others.length === 0) return;

	const damped = new Map(
		others.map((id) => [
			id,
			(pagerank.get(id) ?? 0) /
				Math.max(1, subgraph.degrees?.get(id) ?? 1) ** HUB_DAMPING,
		]),
	);
	const graphScore = normalized(damped);
	const similarity = questionVector
		? normalized(
				new Map(
					Object.entries(await source.similarities(others, questionVector)),
				),
			)
		: null;
	const score = (id: string) =>
		similarity
			? (1 - SIMILARITY_WEIGHT) * (graphScore.get(id) ?? 0) +
				SIMILARITY_WEIGHT * (similarity.get(id) ?? 0)
			: (graphScore.get(id) ?? 0);

	const ranked = others
		.filter((id) => subgraph.nodes.get(id)?.status === "context")
		.sort((a, b) => score(b) - score(a) || compareText(a, b));
	const kept = new Set(ranked.slice(0, keep));
	for (const id of others) {
		const node = subgraph.nodes.get(id);
		if (!node) continue;
		node.score = Math.round(score(id) * 1e6) / 1e6;
		if (node.status === "context" && !kept.has(id)) {
			node.status = "candidate";
			node.pruned_reason = "rank_cutoff";
		}
	}
	const isContext = (id: string) =>
		seeds.has(id) || subgraph.nodes.get(id)?.status === "context";
	for (const edge of subgraph.edges.values()) {
		if (
			edge.status === "context" &&
			!(isContext(edge.source) && isContext(edge.target))
		) {
			edge.status = "candidate";
			edge.pruned_reason = "rank_cutoff";
		}
	}
}
