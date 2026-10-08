import { compareText } from "./graphSource.ts";

export interface WeightedEdge {
	id: string;
	source: string;
	target: string;
	weight: number;
}

// Power iteration over an undirected weighted graph, restarting at the seeds. Nodes and
// edges are walked in id order so every runtime sums in the same order and gets
// bit-identical scores.
export function personalizedPageRank(
	nodeIds: string[],
	edges: WeightedEdge[],
	seeds: string[],
	damping = 0.85,
	maxIterations = 100,
	tolerance = 1e-10,
): Map<string, number> {
	const ids = [...new Set(nodeIds)].sort(compareText);
	const index = new Map(ids.map((id, position) => [id, position]));
	const size = ids.length;
	const neighbours: [number, number][][] = Array.from(
		{ length: size },
		() => [],
	);
	const strength = new Float64Array(size);
	for (const edge of [...edges].sort((a, b) => compareText(a.id, b.id))) {
		const from = index.get(edge.source);
		const to = index.get(edge.target);
		if (from === undefined || to === undefined || from === to) continue;
		neighbours[from]?.push([to, edge.weight]);
		neighbours[to]?.push([from, edge.weight]);
		strength[from] = (strength[from] ?? 0) + edge.weight;
		strength[to] = (strength[to] ?? 0) + edge.weight;
	}
	const seedPositions = seeds.flatMap((id) => {
		const position = index.get(id);
		return position === undefined ? [] : [position];
	});
	const restart = new Float64Array(size);
	for (const position of seedPositions)
		restart[position] = 1 / seedPositions.length;
	if (seedPositions.length === 0) restart.fill(size ? 1 / size : 0);

	let rank = Float64Array.from(restart);
	for (let iteration = 0; iteration < maxIterations; iteration++) {
		const next = new Float64Array(size);
		let dangling = 0;
		for (let from = 0; from < size; from++) {
			const total = strength[from] ?? 0;
			const mass = rank[from] ?? 0;
			if (total === 0) {
				dangling += mass;
				continue;
			}
			for (const [to, weight] of neighbours[from] ?? []) {
				next[to] = (next[to] ?? 0) + (damping * mass * weight) / total;
			}
		}
		let change = 0;
		for (let position = 0; position < size; position++) {
			const restartMass = restart[position] ?? 0;
			const value =
				(next[position] ?? 0) +
				(1 - damping + damping * dangling) * restartMass;
			change += Math.abs(value - (rank[position] ?? 0));
			next[position] = value;
		}
		rank = next;
		if (change < tolerance) break;
	}
	return new Map(ids.map((id, position) => [id, rank[position] ?? 0]));
}
