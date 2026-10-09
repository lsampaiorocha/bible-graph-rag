import { describe, expect, it } from "vitest";
import {
	executeCall,
	type LlmClient,
	MemoryGraphSource,
	planByKeywords,
	runAgent,
	type Subgraph,
} from "../src/index.ts";

const entity = (
	id: string,
	name: string,
	kind: string,
	overviewRank?: number,
) => ({
	id,
	name,
	kind,
	mentionCount: 1,
	...(overviewRank ? { overviewRank } : {}),
});
const link = (id: string, type: string, source: string, target: string) => ({
	id,
	type,
	source,
	target,
	supportCount: 1,
	description: "",
	evidenceVerses: [],
});

function source() {
	return new MemoryGraphSource({
		indexId: "test-index",
		entities: [
			entity("per_jacob", "Jacob", "person", 1),
			entity("per_joseph", "Joseph", "person", 2),
			entity("per_dinah", "Dinah", "person"),
			entity("evt_flood", "The flood", "event", 1),
			entity("evt_sold", "Joseph sold into Egypt", "event", 2),
			entity("evt_dream", "Joseph's dream", "event"),
			entity("per_noah", "Noah", "person"),
		],
		edges: [
			link("r1", "FATHER_OF", "per_jacob", "per_joseph"),
			link("r2", "FATHER_OF", "per_jacob", "per_dinah"),
			link("r3", "PARTICIPATED_IN", "per_noah", "evt_flood"),
			link("r4", "PARTICIPATED_IN", "per_joseph", "evt_sold"),
			link("r5", "PARTICIPATED_IN", "per_joseph", "evt_dream"),
		],
		aliases: [],
	});
}

describe("overview tool", () => {
	it("returns the ranked main characters with only the links among them", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source(),
			{ tool: "overview", entity: "people" },
			1,
			subgraph,
		);

		expect(step).toEqual({
			index: 1,
			tool: "overview",
			args: { entity: "people" },
			node_ids: ["per_jacob", "per_joseph"],
			edge_ids: ["r1"],
			ms: expect.any(Number),
		});
		expect([...subgraph.nodes.keys()]).toEqual(["per_jacob", "per_joseph"]);
	});

	it("returns the ranked story events with the main characters who took part", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source(),
			{ tool: "overview", entity: "events" },
			1,
			subgraph,
		);

		expect(step.node_ids).toEqual(["evt_flood", "evt_sold", "per_joseph"]);
		expect(step.edge_ids).toEqual(["r4"]);
	});

	it("is what the keyword planner picks for main-character and main-event questions", () => {
		expect(
			planByKeywords("Who are the main characters of Genesis?", []),
		).toEqual([{ tool: "overview", entity: "people" }]);
		expect(
			planByKeywords("What are the most important events of Genesis?", []),
		).toEqual([{ tool: "overview", entity: "events" }]);
	});
});

describe("questions that name nobody", () => {
	async function plannerPrompt(question: string) {
		const prompts: string[] = [];
		const outputs: unknown[] = [
			{ reasoning: "", calls: [] },
			{ sentences: [], insufficient_context: true },
		];
		const llm: LlmClient = {
			async complete<T>(request: { prompt: string }) {
				prompts.push(request.prompt);
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
		const graph = new MemoryGraphSource({
			indexId: "test-index",
			entities: [
				{ ...entity("per_noah", "Noah", "person"), embedding: [1, 0] },
			],
			edges: [],
			aliases: [],
		});
		await runAgent(graph, question, llm, async () => [1, 0]);
		return prompts[0] ?? "";
	}

	it("start from the entity nearest in meaning", async () => {
		expect(await plannerPrompt("Who built the ark?")).toContain(
			"per_noah: Noah",
		);
	});

	it("except whole-book questions, where a near entity only lures the planner away", async () => {
		expect(
			await plannerPrompt("What are the most important events of Genesis?"),
		).toContain("No entity names were recognized in the question.");
	});
});

describe("links among retrieved people", () => {
	it("are added, so a wife fetched as a spouse links to the children fetched as children", async () => {
		const outputs: unknown[] = [
			{
				reasoning: "children and wives",
				calls: [
					{
						tool: "relatives",
						entity: "per_jacob",
						relation: "children",
						generations: null,
						other: null,
						family_only: null,
					},
					{
						tool: "relatives",
						entity: "per_jacob",
						relation: "spouses",
						generations: null,
						other: null,
						family_only: null,
					},
				],
			},
			{
				sentences: [{ text: "Leah bore Reuben.", cites: ["R1"] }],
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
		const graph = new MemoryGraphSource({
			indexId: "test-index",
			entities: [
				entity("per_jacob", "Jacob", "person"),
				entity("per_leah", "Leah", "person"),
				entity("per_reuben", "Reuben", "person"),
			],
			edges: [
				link("r_father", "FATHER_OF", "per_jacob", "per_reuben"),
				link("r_spouse", "SPOUSE_OF", "per_jacob", "per_leah"),
				link("r_mother", "MOTHER_OF", "per_leah", "per_reuben"),
			],
			aliases: [],
		});

		const trace = await runAgent(
			graph,
			"Who were Jacob's children and their mothers?",
			llm,
		);

		expect(trace.edges.map((edge) => edge.id).sort()).toEqual([
			"r_father",
			"r_mother",
			"r_spouse",
		]);
	});
});
