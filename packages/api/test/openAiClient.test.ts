import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OpenAiCompatibleClient } from "../src/openAiClient.ts";

describe("OpenAiCompatibleClient", () => {
	afterEach(() => vi.unstubAllGlobals());

	it("sends the schema, parses fenced JSON, prices the tokens and logs the call", async () => {
		const requests: { url: string; body: unknown; auth: string | null }[] = [];
		vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
			requests.push({
				url,
				body: JSON.parse(String(init.body)),
				auth: new Headers(init.headers).get("Authorization"),
			});
			return new Response(
				JSON.stringify({
					choices: [{ message: { content: '```json\n{"word":"OK"}\n```' } }],
					usage: { prompt_tokens: 1000, completion_tokens: 200 },
				}),
			);
		});
		const ledgerPath = join(
			mkdtempSync(join(tmpdir(), "kgrag-")),
			"ledger.jsonl",
		);
		const client = new OpenAiCompatibleClient({
			baseUrl: "http://localhost:11434/v1/",
			apiKey: "secret",
			model: "qwen3.5:4b",
			maxCallUsd: 0.1,
			projectBudgetUsd: 40,
			priceInputPerMtok: 0.5,
			priceOutputPerMtok: 2,
			ledgerPath,
		});

		const result = await client.complete<{ word: string }>({
			purpose: "agent_answer",
			system: "Be brief.",
			prompt: "Say OK.",
			schema: { type: "object" },
		});

		expect(requests).toEqual([
			{
				url: "http://localhost:11434/v1/chat/completions",
				auth: "Bearer secret",
				body: {
					model: "qwen3.5:4b",
					messages: [
						{ role: "system", content: "Be brief." },
						{ role: "user", content: "Say OK." },
					],
					response_format: {
						type: "json_schema",
						json_schema: {
							name: "output",
							schema: { type: "object" },
							strict: false,
						},
					},
				},
			},
		]);
		expect(result).toEqual({
			output: { word: "OK" },
			usage: {
				model: "qwen3.5:4b",
				input_tokens: 1000,
				output_tokens: 200,
				cost_usd: 0.0009,
			},
		});
		expect(JSON.parse(readFileSync(ledgerPath, "utf-8"))).toMatchObject({
			purpose: "agent_answer",
			model: "qwen3.5:4b",
			cost_usd: 0.0009,
		});
	});

	it("refuses to call once the project budget is spent", async () => {
		const client = new OpenAiCompatibleClient({
			baseUrl: "http://localhost:11434/v1",
			apiKey: "none",
			model: "qwen3.5:4b",
			maxCallUsd: 0.1,
			projectBudgetUsd: 0.05,
			priceInputPerMtok: 0,
			priceOutputPerMtok: 0,
			ledgerPath: join(mkdtempSync(join(tmpdir(), "kgrag-")), "ledger.jsonl"),
		});

		await expect(
			client.complete({
				purpose: "agent_answer",
				system: "",
				prompt: "",
				schema: {},
			}),
		).rejects.toThrow("project budget reached: $0.00 of $0.05");
	});
});
