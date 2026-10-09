import { describe, expect, it } from "vitest";
import { type LlmClient, MemoryGraphSource, runVector } from "../src/index.ts";

describe("runVector", () => {
	it("counts a verse marker cited from inside a given passage as citing that passage", async () => {
		const source = new MemoryGraphSource({
			indexId: "test-index",
			entities: [],
			edges: [],
			aliases: [],
			passages: [
				{
					id: "Gen.4.p2",
					text: "[Gen.4.18] And unto Enoch was born Irad: and Irad begat Mehujael.",
					verseRefs: ["Gen.4.18"],
					embedding: [1, 0],
				},
			],
		});
		const llm: LlmClient = {
			async complete<T>() {
				return {
					output: {
						sentences: [
							{
								text: "Irad was the father of Mehujael.",
								cites: ["Gen.4.18", "Gen.9.9"],
							},
						],
						insufficient_context: false,
					} as T,
					usage: {
						model: "fake-model",
						input_tokens: 10,
						output_tokens: 5,
						cost_usd: 0.001,
					},
				};
			},
		};

		const trace = await runVector(
			source,
			"Who was the father of Mehujael?",
			llm,
			async () => [1, 0],
		);

		expect(trace.answer).toEqual({
			text: "Irad was the father of Mehujael.",
			sentences: [
				{ text: "Irad was the father of Mehujael.", cites: ["Gen.4.p2"] },
			],
			invalid_handles: ["Gen.9.9"],
			uncited_sentence_idx: [],
			insufficient_context: false,
		});
	});
});
