import { compareText, KIN_TYPES, type LinkRow } from "./graphSource.ts";

export const PATH_COUNT = 3;
export const PATH_MAX_LINKS = 4;
// Cousins and in-laws sit 4 to 6 parent, sibling and spouse links apart.
export const FAMILY_PATH_MAX_LINKS = 6;
// Every entity a path passes through adds HUB_WEIGHT * log2(1 + its links in the
// corridor) to the path's cost, so a chain through God or a crowded event ranks below a
// specific one of the same length.
export const HUB_WEIGHT = 0.5;

export interface FoundPath {
	cost: number;
	nodeIds: string[];
	edgeIds: string[];
}

interface Step {
	neighbour: string;
	edgeId: string;
}

const pairKey = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`);
const kinRank = (type: string) =>
	(KIN_TYPES as readonly string[]).includes(type) ? 0 : 1;

// Yen's k shortest simple paths, with ties broken by the node ids along the path so
// Neo4j and the browser pick the same paths.
export function shortestPaths(
	rows: LinkRow[],
	from: string,
	to: string,
	maxLinks: number,
	count = PATH_COUNT,
): FoundPath[] {
	// One link per pair of entities, a family link when there is one.
	const preferred = (a: LinkRow, b: LinkRow) =>
		kinRank(a.type) - kinRank(b.type) || compareText(a.edgeId, b.edgeId);
	const chosen = new Map<string, LinkRow>();
	for (const row of rows) {
		const key = pairKey(row.source, row.target);
		const current = chosen.get(key);
		if (!current || preferred(row, current) < 0) chosen.set(key, row);
	}
	const adjacency = new Map<string, Step[]>();
	for (const row of chosen.values()) {
		for (const [a, b] of [
			[row.source, row.target],
			[row.target, row.source],
		] as const) {
			adjacency.set(a, [
				...(adjacency.get(a) ?? []),
				{ neighbour: b, edgeId: row.edgeId },
			]);
		}
	}
	for (const steps of adjacency.values())
		steps.sort((x, y) => compareText(x.neighbour, y.neighbour));
	const passCost = (id: string) =>
		HUB_WEIGHT * Math.log2(1 + (adjacency.get(id)?.length ?? 0));
	const costOf = (nodeIds: string[]) =>
		nodeIds.length -
		1 +
		nodeIds.slice(1, -1).reduce((sum, id) => sum + passCost(id), 0);
	const edgeBetween = (a: string, b: string) =>
		(chosen.get(pairKey(a, b)) as LinkRow).edgeId;
	const better = (
		a: { cost: number; key: string },
		b?: { cost: number; key: string },
	) =>
		!b ||
		a.cost < b.cost - 1e-9 ||
		(Math.abs(a.cost - b.cost) <= 1e-9 && compareText(a.key, b.key) < 0);

	// Cheapest path within a link budget: costs are positive, so the best walk found by
	// relaxing one link at a time never repeats a node.
	function cheapest(
		start: string,
		blockedNodes: Set<string>,
		blockedPairs: Set<string>,
		budget: number,
	): string[] | null {
		let layer = new Map([[start, { cost: 0, key: start, nodes: [start] }]]);
		let best: { cost: number; key: string; nodes: string[] } | undefined;
		for (let links = 1; links <= budget && layer.size > 0; links++) {
			const next = new Map<
				string,
				{ cost: number; key: string; nodes: string[] }
			>();
			for (const [id, state] of [...layer].sort(([a], [b]) =>
				compareText(a, b),
			)) {
				for (const { neighbour } of adjacency.get(id) ?? []) {
					if (
						blockedNodes.has(neighbour) ||
						blockedPairs.has(pairKey(id, neighbour)) ||
						state.nodes.includes(neighbour)
					)
						continue;
					const nodes = [...state.nodes, neighbour];
					const candidate = {
						cost: state.cost + 1 + (neighbour === to ? 0 : passCost(neighbour)),
						key: nodes.join(">"),
						nodes,
					};
					if (neighbour === to) {
						if (better(candidate, best)) best = candidate;
					} else if (better(candidate, next.get(neighbour))) {
						next.set(neighbour, candidate);
					}
				}
			}
			layer = next;
		}
		return best?.nodes ?? null;
	}

	const first = cheapest(from, new Set(), new Set(), maxLinks);
	if (!first) return [];
	const found: string[][] = [first];
	const pending = new Map<string, string[]>();
	while (found.length < count) {
		const previous = found[found.length - 1] as string[];
		for (let i = 0; i < previous.length - 1; i++) {
			const rootPath = previous.slice(0, i + 1);
			const blockedPairs = new Set(
				found
					.filter(
						(path) => path.slice(0, i + 1).join(">") === rootPath.join(">"),
					)
					.map((path) => pairKey(path[i] as string, path[i + 1] as string)),
			);
			const spur = cheapest(
				previous[i] as string,
				new Set(rootPath.slice(0, -1)),
				blockedPairs,
				maxLinks - i,
			);
			if (spur) {
				const nodes = [...rootPath.slice(0, -1), ...spur];
				pending.set(nodes.join(">"), nodes);
			}
		}
		for (const path of found) pending.delete(path.join(">"));
		const next = [...pending.values()].sort(
			(a, b) => costOf(a) - costOf(b) || compareText(a.join(">"), b.join(">")),
		)[0];
		if (!next) break;
		pending.delete(next.join(">"));
		found.push(next);
	}
	return found.map((nodeIds) => ({
		cost: Math.round(costOf(nodeIds) * 1e4) / 1e4,
		nodeIds,
		edgeIds: nodeIds
			.slice(1)
			.map((id, i) => edgeBetween(nodeIds[i] as string, id)),
	}));
}
