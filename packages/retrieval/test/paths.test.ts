import { describe, expect, it } from "vitest";
import {
	executeCall,
	KIN_TYPES,
	type LlmClient,
	MemoryGraphSource,
	planByKeywords,
	runAgent,
	type SeedMatch,
	type Subgraph,
	shortestPaths,
} from "../src/index.ts";

const entity = (id: string, name: string, kind = "person") => ({
	id,
	name,
	kind,
	mentionCount: 1,
});
const link = (id: string, type: string, source: string, target: string) => ({
	id,
	type,
	source,
	target,
	supportCount: 1,
	description: `${source} ${type} ${target}`,
	evidenceVerses: [],
});

// Lot and Abram meet three ways: a shared event, God (who speaks to many), and family.
function source() {
	return new MemoryGraphSource({
		indexId: "test-index",
		entities: [
			entity("per_lot", "Lot"),
			entity("per_haran", "Haran"),
			entity("per_terah", "Terah"),
			entity("per_abram", "Abram"),
			entity("per_god", "God"),
			entity("evt_parting", "Abram and Lot part", "event"),
			...["a", "b", "c", "d", "e"].map((suffix) =>
				entity(`per_listener_${suffix}`, `Listener ${suffix}`),
			),
		],
		edges: [
			link("r1", "FATHER_OF", "per_haran", "per_lot"),
			link("r2", "FATHER_OF", "per_terah", "per_haran"),
			link("r3", "FATHER_OF", "per_terah", "per_abram"),
			link("r4", "PARTICIPATED_IN", "per_lot", "evt_parting"),
			link("r5", "PARTICIPATED_IN", "per_abram", "evt_parting"),
			link("r6", "SPOKE_TO", "per_god", "per_abram"),
			link("r7", "SPOKE_TO", "per_god", "per_lot"),
			...["a", "b", "c", "d", "e"].map((suffix) =>
				link(`r_${suffix}`, "SPOKE_TO", "per_god", `per_listener_${suffix}`),
			),
		],
		aliases: [
			{ key: "lot", nodeId: "per_lot" },
			{ key: "abram", nodeId: "per_abram" },
		],
	});
}

describe("shortestPaths", () => {
	it("ranks a shared event above the God hub, and both above the longer family chain", async () => {
		const rows = await source().corridor(["per_lot", "per_abram"], 2, null);

		expect(shortestPaths(rows, "per_lot", "per_abram", 4)).toEqual([
			{
				cost: 2.7925,
				nodeIds: ["per_lot", "evt_parting", "per_abram"],
				edgeIds: ["r4", "r5"],
			},
			{
				cost: 3.5,
				nodeIds: ["per_lot", "per_god", "per_abram"],
				edgeIds: ["r7", "r6"],
			},
			{
				cost: 4.585,
				nodeIds: ["per_lot", "per_haran", "per_terah", "per_abram"],
				edgeIds: ["r1", "r2", "r3"],
			},
		]);
	});

	it("follows only family links when the corridor holds only family links", async () => {
		const rows = await source().corridor(
			["per_lot", "per_abram"],
			3,
			KIN_TYPES,
		);

		expect(shortestPaths(rows, "per_lot", "per_abram", 6)).toEqual([
			{
				cost: 4.585,
				nodeIds: ["per_lot", "per_haran", "per_terah", "per_abram"],
				edgeIds: ["r1", "r2", "r3"],
			},
		]);
	});

	it("returns nothing when no path fits the link budget", async () => {
		const rows = await source().corridor(
			["per_lot", "per_abram"],
			3,
			KIN_TYPES,
		);

		expect(shortestPaths(rows, "per_lot", "per_abram", 2)).toEqual([]);
	});
});

describe("path tool", () => {
	it("adds the family chain to the subgraph and records it as a ranked path", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source(),
			{
				tool: "path",
				entity: "per_lot",
				other: "per_abram",
				family_only: true,
			},
			1,
			subgraph,
		);

		expect(step).toEqual({
			index: 1,
			tool: "path",
			args: { entity: "per_lot", other: "per_abram", family_only: "yes" },
			entity_id: "per_lot",
			node_ids: ["per_abram", "per_haran", "per_lot", "per_terah"],
			edge_ids: ["r1", "r2", "r3"],
			ms: expect.any(Number),
		});
		expect(subgraph.paths).toEqual([
			{
				rank: 1,
				cost: 4.585,
				status: "context",
				node_ids: ["per_lot", "per_haran", "per_terah", "per_abram"],
				edge_ids: ["r1", "r2", "r3"],
			},
		]);
	});

	it("is what the keyword planner picks for a kinship question naming two people", () => {
		const seed = (nodeId: string): SeedMatch => ({
			seed: { node_id: nodeId, method: "alias_exact", score: 1 },
			label: nodeId,
		});

		expect(
			planByKeywords("How is Lot related to Abram?", [
				seed("per_lot"),
				seed("per_abram"),
			]),
		).toEqual([
			{
				tool: "path",
				entity: "per_lot",
				other: "per_abram",
				family_only: true,
			},
		]);
	});

	it("fills a missing second entity with the other entity named in the question", async () => {
		const outputs: unknown[] = [
			{
				reasoning: "kinship",
				calls: [
					{
						tool: "path",
						entity: "per_lot",
						relation: null,
						generations: null,
						other: null,
						family_only: true,
					},
				],
			},
			{
				sentences: [
					{ text: "Haran, Lot's father, was a son of Terah.", cites: ["R1"] },
				],
				insufficient_context: false,
			},
		];
		const llm: LlmClient = {
			async complete<T>() {
				return {
					output: outputs.shift() as T,
					usage: {
						model: "fake-model",
						input_tokens: 10,
						output_tokens: 5,
						cost_usd: 0.001,
					},
				};
			},
		};

		const trace = await runAgent(source(), "How is Lot related to Abram?", llm);

		expect(trace.steps?.map((step) => step.args)).toEqual([
			{ entity: "per_lot", other: "per_abram", family_only: "yes" },
		]);
		expect(trace.paths.map((path) => path.node_ids)).toEqual([
			["per_lot", "per_haran", "per_terah", "per_abram"],
		]);
	});
});
