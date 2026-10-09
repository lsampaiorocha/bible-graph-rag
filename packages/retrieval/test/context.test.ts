import { describe, expect, it } from "vitest";
import { applyAnswer } from "../src/index.ts";

describe("applyAnswer", () => {
	it("takes handles written inside a sentence out of its text and counts them as citations", () => {
		const answer = applyAnswer(
			{
				sentences: [
					{ text: "The flood destroys all flesh. [E8, R42]", cites: ["C1"] },
				],
				insufficient_context: false,
			},
			{ E8: "evt_flood", R42: "rel_ark", C1: "Gen.7.21" },
			[],
			[],
			[],
		);

		expect(answer).toEqual({
			text: "The flood destroys all flesh.",
			sentences: [
				{
					text: "The flood destroys all flesh.",
					cites: ["Gen.7.21", "evt_flood", "rel_ark"],
				},
			],
			invalid_handles: [],
			uncited_sentence_idx: [],
			insufficient_context: false,
		});
	});
});
