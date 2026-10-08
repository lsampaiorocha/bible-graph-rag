import { describe, expect, it } from "vitest";
import { aliasKey, MemoryGraphSource, runLocal } from "../src/index.ts";

describe("aliasKey", () => {
	it("produces the same keys as the Python loader for titles, hyphens and possessives", () => {
		expect(
			["the LORD", "Beth-el", "Rebekah’s", "Lord GOD", "  Padan-aram "].map(
				aliasKey,
			),
		).toEqual(["lord", "bethel", "rebekah", "lord god", "padanaram"]);
	});
});

describe("runLocal", () => {
	it("keeps the top_m neighbours by support_count and marks the next top_m as neighbour_cap candidates", async () => {
		const names = [
			"noah",
			"shem",
			"ham",
			"japheth",
			"god",
			"ark",
			"canaan",
			"arphaxad",
			"cush",
			"mizraim",
		];
		const link = (id: string, a: string, b: string, supportCount: number) => ({
			id,
			type: "FATHER_OF",
			source: `per_${a}`,
			target: `per_${b}`,
			supportCount,
		});
		const source = new MemoryGraphSource({
			indexId: "test-index",
			entities: names.map((name) => ({
				id: `per_${name}`,
				name,
				mentionCount: 1,
			})),
			edges: [
				link("r1", "noah", "shem", 3),
				link("r2", "noah", "ham", 3),
				link("r3", "noah", "japheth", 2),
				link("r4", "god", "noah", 2),
				link("r5", "noah", "ark", 1),
				link("r6", "ham", "canaan", 2),
				link("r7", "shem", "arphaxad", 1),
				link("r8", "ham", "cush", 1),
				link("r9", "ham", "mizraim", 1),
			],
			aliases: [{ key: "noah", nodeId: "per_noah" }],
		});

		const { trace_id, created_at, metrics, ...trace } = await runLocal(
			source,
			"Who were the sons of Noah?",
			{
				maxHops: 2,
				topM: 2,
			},
		);

		const node = (
			name: string,
			hop: number,
			status: "context" | "candidate",
			extra: object = {},
		) => ({
			id: `per_${name}`,
			label: name,
			hop,
			score: 1 / (1 + hop),
			status,
			...extra,
		});
		const edge = (
			id: string,
			a: string,
			b: string,
			hop: number,
			score: number,
			extra: object = {},
		) => ({
			id,
			source: `per_${a}`,
			target: `per_${b}`,
			type: "FATHER_OF",
			hop,
			score,
			evidence_chunk_ids: [],
			...extra,
		});
		const capped = { status: "candidate", pruned_reason: "neighbour_cap" };
		expect(trace).toEqual({
			trace_version: "1.0",
			index_id: "test-index",
			source: "live",
			query: {
				text: "Who were the sons of Noah?",
				mode: "local",
				params: { max_hops: 2, top_m: 2 },
			},
			seeds: [
				{
					node_id: "per_noah",
					method: "alias_exact",
					matched_text: "noah",
					score: 1,
				},
			],
			nodes: [
				node("noah", 0, "context", { handle: "E1" }),
				node("god", 1, "candidate", {
					reached_via: "r4",
					pruned_reason: "neighbour_cap",
				}),
				node("ham", 1, "context", { reached_via: "r2", handle: "E2" }),
				node("japheth", 1, "candidate", {
					reached_via: "r3",
					pruned_reason: "neighbour_cap",
				}),
				node("shem", 1, "context", { reached_via: "r1", handle: "E3" }),
				node("arphaxad", 2, "context", { reached_via: "r7", handle: "E4" }),
				node("canaan", 2, "context", { reached_via: "r6", handle: "E5" }),
				node("cush", 2, "candidate", {
					reached_via: "r8",
					pruned_reason: "neighbour_cap",
				}),
				node("mizraim", 2, "candidate", {
					reached_via: "r9",
					pruned_reason: "neighbour_cap",
				}),
			],
			edges: [
				edge("r1", "noah", "shem", 1, 1, { status: "context", handle: "R1" }),
				edge("r2", "noah", "ham", 1, 1, { status: "context", handle: "R2" }),
				edge("r3", "noah", "japheth", 1, 2 / 3, capped),
				edge("r4", "god", "noah", 1, 2 / 3, capped),
				edge("r6", "ham", "canaan", 2, 2 / 3, {
					status: "context",
					handle: "R3",
				}),
				edge("r7", "shem", "arphaxad", 2, 1 / 3, {
					status: "context",
					handle: "R4",
				}),
				edge("r8", "ham", "cush", 2, 1 / 3, capped),
				edge("r9", "ham", "mizraim", 2, 1 / 3, capped),
			],
			paths: [],
			communities: [],
			chunks: [],
			context: {
				handles: {
					E1: "per_noah",
					E2: "per_ham",
					E3: "per_shem",
					E4: "per_arphaxad",
					E5: "per_canaan",
					R1: "r1",
					R2: "r2",
					R3: "r6",
					R4: "r7",
				},
				token_count: 0,
				token_budget: 0,
				truncated: false,
			},
			answer: null,
		});
		expect([
			typeof trace_id,
			typeof created_at,
			Object.keys(metrics.timings_ms),
		]).toEqual(["string", "string", ["seed", "expand"]]);
	});
});
