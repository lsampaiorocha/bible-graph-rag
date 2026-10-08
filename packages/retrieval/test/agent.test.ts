import { describe, expect, it } from "vitest";
import {
	type LlmClient,
	MemoryGraphSource,
	planByKeywords,
	runAgent,
	type SeedMatch,
} from "../src/index.ts";

const person = (id: string, name: string, firstVerse: string) => ({
	id,
	name,
	mentionCount: 3,
	kind: "person",
	gender: "male",
	description: `${name}, first named ${firstVerse}.`,
	firstVerse,
});
const fatherOf = (
	id: string,
	source: string,
	target: string,
	verse: string,
) => ({
	id,
	type: "FATHER_OF",
	source,
	target,
	supportCount: 1,
	description: `${source} begat ${target}.`,
	evidenceVerses: [verse],
});

function genesisSource() {
	return new MemoryGraphSource({
		indexId: "test-index",
		entities: [
			person("per_methusael", "Methusael", "Gen.4.18"),
			person("per_lamech_gen_4_18", "Lamech", "Gen.4.18"),
			person("per_methuselah", "Methuselah", "Gen.5.21"),
			person("per_lamech_gen_5_25", "Lamech", "Gen.5.25"),
			person("per_noah", "Noah", "Gen.5.29"),
		],
		edges: [
			fatherOf("r_a", "per_methusael", "per_lamech_gen_4_18", "Gen.4.18"),
			fatherOf("r_b", "per_methuselah", "per_lamech_gen_5_25", "Gen.5.25"),
			fatherOf("r_c", "per_lamech_gen_5_25", "per_noah", "Gen.5.29"),
		],
		aliases: [
			{ key: "noah", nodeId: "per_noah" },
			{ key: "lamech", nodeId: "per_lamech_gen_4_18" },
			{ key: "lamech", nodeId: "per_lamech_gen_5_25" },
		],
		verses: {
			"Gen.4.18": "Methusael begat Lamech.",
			"Gen.5.25":
				"Methuselah lived an hundred eighty and seven years, and begat Lamech.",
			"Gen.5.29": "And he called his name Noah.",
		},
	});
}

function scriptedLlm(outputs: unknown[]): LlmClient & { prompts: string[] } {
	const prompts: string[] = [];
	return {
		prompts,
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
}

describe("planByKeywords", () => {
	const seed = (id: string): SeedMatch => ({
		seed: { node_id: id, method: "alias_exact", matched_text: id, score: 1 },
		label: id,
	});

	it("maps family words to the matching tool and anything else to explore", () => {
		expect(
			planByKeywords("Who were the ancestors of Noah?", [seed("per_noah")]),
		).toEqual([{ tool: "ancestors", entity: "per_noah" }]);
		expect(
			planByKeywords("Who were the sons of Noah?", [seed("per_noah")]),
		).toEqual([{ tool: "relatives", entity: "per_noah", relation: "sons" }]);
		expect(planByKeywords("Who did Jacob marry?", [seed("per_jacob")])).toEqual(
			[{ tool: "relatives", entity: "per_jacob", relation: "spouses" }],
		);
		expect(
			planByKeywords("What happened at Bethel?", [seed("pla_bethel")]),
		).toEqual([{ tool: "explore", entity: "pla_bethel" }]);
	});
});

describe("runAgent", () => {
	it("answers a lineage question without an LLM by following parent links only", async () => {
		const trace = await runAgent(
			genesisSource(),
			"Who were the ancestors of Noah?",
			null,
		);

		expect(trace.steps?.map(({ ms, ...step }) => step)).toEqual([
			{
				index: 1,
				tool: "ancestors",
				args: { entity: "per_noah" },
				entity_id: "per_noah",
				node_ids: ["per_lamech_gen_5_25", "per_methuselah", "per_noah"],
				edge_ids: ["r_b", "r_c"],
			},
		]);
		expect(
			trace.nodes.map((node) => [node.id, node.hop, node.status, node.handle]),
		).toEqual([
			["per_noah", 0, "context", "E1"],
			["per_lamech_gen_5_25", 1, "context", "E2"],
			["per_methuselah", 2, "context", "E3"],
		]);
		expect(trace.answer).toBeNull();
		expect(trace.query.params).toEqual({
			max_tool_calls: 3,
			planner: "keywords",
		});
	});

	it("validates the LLM's citations and promotes what the answer used", async () => {
		const llm = scriptedLlm([
			{
				reasoning: "lineage",
				calls: [
					{
						tool: "ancestors",
						entity: "per_noah",
						relation: null,
						generations: null,
					},
				],
			},
			{
				sentences: [
					{ text: "Noah's father was Lamech.", cites: ["[R1]", "C1"] },
					{ text: "Lamech's father was Methuselah.", cites: ["R2", "R99"] },
					{ text: "That is the whole line.", cites: [] },
				],
				insufficient_context: false,
			},
		]);

		const trace = await runAgent(
			genesisSource(),
			"Who were Noah's forefathers?",
			llm,
		);

		expect(trace.answer).toEqual({
			text: "Noah's father was Lamech. Lamech's father was Methuselah. That is the whole line.",
			sentences: [
				{ text: "Noah's father was Lamech.", cites: ["r_c", "Gen.5.29"] },
				{ text: "Lamech's father was Methuselah.", cites: ["r_b"] },
				{ text: "That is the whole line.", cites: [] },
			],
			invalid_handles: ["R99"],
			uncited_sentence_idx: [2],
			insufficient_context: false,
		});
		expect(
			trace.edges.map((edge) => [edge.id, edge.handle, edge.status]),
		).toEqual([
			["r_c", "R1", "cited"],
			["r_b", "R2", "cited"],
		]);
		expect(
			trace.chunks.map((chunk) => [chunk.id, chunk.handle, chunk.status]),
		).toEqual([
			["Gen.5.29", "C1", "cited"],
			["Gen.5.25", "C2", "context"],
		]);
		expect(trace.metrics.usage).toEqual({
			model: "fake-model",
			input_tokens: 20,
			output_tokens: 10,
			cost_usd: 0.002,
		});
	});

	it("plans a second round when a name matches several people, and uses the id it picks", async () => {
		const llm = scriptedLlm([
			{
				reasoning: "lineage",
				calls: [
					{
						tool: "ancestors",
						entity: "Lamech",
						relation: null,
						generations: null,
					},
				],
			},
			{
				reasoning: "the Lamech who fathered Noah",
				calls: [
					{
						tool: "ancestors",
						entity: "per_lamech_gen_5_25",
						relation: null,
						generations: null,
					},
				],
			},
			{
				sentences: [{ text: "Lamech's father was Methuselah.", cites: ["R1"] }],
				insufficient_context: false,
			},
		]);

		const trace = await runAgent(
			genesisSource(),
			"Who was the father of the Lamech who begat Noah?",
			llm,
		);

		expect(
			trace.steps?.map((step) => [
				step.tool,
				step.entity_id,
				step.note ?? null,
			]),
		).toEqual([
			["ancestors", "per_lamech_gen_4_18", '2 entities are named "Lamech"'],
			["ancestors", "per_lamech_gen_5_25", null],
		]);
		expect(llm.prompts[1]).toContain(
			"candidates per_lamech_gen_4_18, per_lamech_gen_5_25",
		);
	});
});
