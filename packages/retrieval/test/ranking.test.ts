import { describe, expect, it } from "vitest";
import {
	executeCall,
	MemoryGraphSource,
	personalizedPageRank,
	rankSubgraph,
	runAgent,
	type Subgraph,
	type TraceEdge,
	type TraceNode,
} from "../src/index.ts";

describe("personalizedPageRank", () => {
	it("gives the same scores whatever order the edges arrive in", () => {
		const edges = [
			{ id: "e1", source: "seed", target: "a", weight: 2 },
			{ id: "e2", source: "a", target: "b", weight: 1 },
			{ id: "e3", source: "seed", target: "c", weight: 1 },
		];

		const forward = personalizedPageRank(["seed", "a", "b", "c"], edges, [
			"seed",
		]);
		const backward = personalizedPageRank(
			["c", "b", "a", "seed"],
			[...edges].reverse(),
			["seed"],
		);

		expect([...backward]).toEqual([...forward]);
		expect(
			[...forward.values()].reduce((sum, score) => sum + score, 0),
		).toBeCloseTo(1, 10);
		expect(forward.get("a")).toBeGreaterThan(forward.get("b") ?? 0);
	});
});

describe("rankSubgraph", () => {
	function star(): Subgraph {
		const node = (id: string, hop: number): TraceNode => ({
			id,
			label: id,
			hop,
			score: 0,
			status: "context",
		});
		const edge = (id: string, target: string): TraceEdge => ({
			id,
			source: "seed",
			target,
			type: "OTHER",
			hop: 1,
			score: 1,
			status: "context",
			evidence_chunk_ids: [],
		});
		return {
			nodes: new Map([
				["seed", node("seed", 0)],
				["hub", node("hub", 1)],
				["local", node("local", 1)],
			]),
			edges: new Map([
				["r_hub", edge("r_hub", "hub")],
				["r_local", edge("r_local", "local")],
			]),
			degrees: new Map([
				["hub", 184],
				["local", 2],
			]),
		};
	}
	const source = new MemoryGraphSource({
		indexId: "test",
		entities: [
			{ id: "hub", name: "God", mentionCount: 300, embedding: [1, 0] },
			{ id: "local", name: "Luz", mentionCount: 3, embedding: [0, 1] },
		],
		edges: [],
		aliases: [],
	});

	it("keeps a local entity over an equally linked hub once degree is taken into account", async () => {
		const subgraph = star();

		await rankSubgraph(source, subgraph, ["seed"], null, 1);

		expect(
			[...subgraph.nodes.values()].map((node) => [
				node.id,
				node.status,
				node.pruned_reason ?? null,
			]),
		).toEqual([
			["seed", "context", null],
			["hub", "candidate", "rank_cutoff"],
			["local", "context", null],
		]);
		expect(subgraph.edges.get("r_hub")?.status).toBe("candidate");
	});

	it("lets similarity to the question outweigh the hub penalty", async () => {
		const subgraph = star();

		await rankSubgraph(source, subgraph, ["seed"], [1, 0], 1);

		expect(subgraph.nodes.get("hub")?.status).toBe("context");
		expect(subgraph.nodes.get("local")?.pruned_reason).toBe("rank_cutoff");
	});
});

describe("matching by meaning", () => {
	const source = new MemoryGraphSource({
		indexId: "test",
		entities: [
			{
				id: "per_methuselah",
				name: "Methuselah",
				mentionCount: 5,
				embedding: [1, 0],
			},
			{
				id: "per_lamech",
				name: "Lamech",
				mentionCount: 5,
				embedding: [0.6, 0.8],
			},
			{
				id: "evt_birth_of_methuselah",
				name: "Birth of Methuselah",
				mentionCount: 1,
				embedding: [1, 0.01],
			},
		],
		edges: [
			{
				id: "r1",
				type: "FATHER_OF",
				source: "per_methuselah",
				target: "per_lamech",
				supportCount: 1,
			},
		],
		aliases: [
			{ key: "methuselah", nodeId: "per_methuselah" },
			{ key: "lamech", nodeId: "per_lamech" },
		],
	});
	const vectors: Record<string, number[]> = {
		Methusela: [0.99, 0.05],
		"tax law": [-1, 0],
	};
	const embed = async (text: string) => vectors[text] ?? [0, 0];

	it("resolves a misspelled name to the nearest person, skipping events", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source,
			{ tool: "descendants", entity: "Methusela" },
			1,
			subgraph,
			{ embed },
		);

		expect([step.entity_id, step.note, step.edge_ids]).toEqual([
			"per_methuselah",
			'no entity named "Methusela"; matched by meaning to Methuselah',
			["r1"],
		]);
	});

	it("refuses a match that is not close enough", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source,
			{ tool: "explore", entity: "tax law" },
			1,
			subgraph,
			{ embed },
		);

		expect([step.entity_id ?? null, step.note]).toEqual([
			null,
			'no entity named "tax law"',
		]);
	});

	it("seeds a question that names nobody from the entities close enough in meaning", async () => {
		const trace = await runAgent(
			source,
			"Who came before the flood?",
			null,
			async () => [1, 0],
		);

		expect(trace.seeds.map((seed) => [seed.node_id, seed.method])).toEqual([
			["per_methuselah", "embedding"],
			["evt_birth_of_methuselah", "embedding"],
		]);
	});
});
