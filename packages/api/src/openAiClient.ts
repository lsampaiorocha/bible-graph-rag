import type { LlmClient, LlmUsage } from "@kgrag/retrieval";
import { appendLedger, LEDGER_PATH, ledgerTotal } from "./ledger.ts";

const CALL_TIMEOUT_MS = 300_000;

export interface OpenAiCompatibleOptions {
	baseUrl: string;
	apiKey: string;
	model: string;
	maxCallUsd: number;
	projectBudgetUsd: number;
	// Servers report tokens, not cost: USD per million tokens, zero for local models.
	priceInputPerMtok: number;
	priceOutputPerMtok: number;
	ledgerPath?: string;
}

interface ChatResponse {
	choices?: { message?: { content?: string | null } }[];
	usage?: { prompt_tokens?: number; completion_tokens?: number };
}

// Some models wrap JSON in a ```json fence even when asked for a schema.
function parseJsonOutput(content: string): unknown {
	const text = content
		.trim()
		.replace(/^```(?:json)?/, "")
		.replace(/```$/, "");
	return JSON.parse(text);
}

// Any server speaking the OpenAI chat API: OpenAI, Ollama, vLLM, LM Studio, OpenRouter.
export class OpenAiCompatibleClient implements LlmClient {
	private readonly options: OpenAiCompatibleOptions;

	constructor(options: OpenAiCompatibleOptions) {
		this.options = options;
	}

	async complete<T>(request: {
		purpose: string;
		system: string;
		prompt: string;
		schema: Record<string, unknown>;
	}): Promise<{ output: T; usage: LlmUsage }> {
		const { baseUrl, apiKey, model, maxCallUsd, projectBudgetUsd } =
			this.options;
		const ledgerPath = this.options.ledgerPath ?? LEDGER_PATH;
		const spent = ledgerTotal(ledgerPath);
		if (spent + maxCallUsd > projectBudgetUsd) {
			throw new Error(
				`project budget reached: $${spent.toFixed(2)} of $${projectBudgetUsd.toFixed(2)}`,
			);
		}
		const started = performance.now();
		const response = await fetch(
			`${baseUrl.replace(/\/$/, "")}/chat/completions`,
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
				},
				body: JSON.stringify({
					model,
					messages: [
						{ role: "system", content: request.system },
						{ role: "user", content: request.prompt },
					],
					response_format: {
						type: "json_schema",
						json_schema: {
							name: "output",
							schema: request.schema,
							strict: false,
						},
					},
				}),
				signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
			},
		);
		if (!response.ok)
			throw new Error(
				`HTTP ${response.status} from ${baseUrl}: ${(await response.text()).slice(0, 300)}`,
			);
		const body = (await response.json()) as ChatResponse;
		const inputTokens = body.usage?.prompt_tokens ?? 0;
		const outputTokens = body.usage?.completion_tokens ?? 0;
		const costUsd =
			(inputTokens * this.options.priceInputPerMtok +
				outputTokens * this.options.priceOutputPerMtok) /
			1e6;
		appendLedger(
			{
				time: `${new Date().toISOString().slice(0, 19)}+00:00`,
				run_id: `api-${new Date().toISOString().slice(0, 10)}`,
				purpose: request.purpose,
				model,
				effort: "",
				item_id: request.prompt.slice(0, 80),
				attempt: 1,
				is_error: false,
				input_tokens: inputTokens,
				output_tokens: outputTokens,
				cost_usd: costUsd,
				duration_ms: Math.round(performance.now() - started),
			},
			ledgerPath,
		);
		return {
			output: parseJsonOutput(body.choices?.[0]?.message?.content ?? "") as T,
			usage: {
				model,
				input_tokens: inputTokens,
				output_tokens: outputTokens,
				cost_usd: costUsd,
			},
		};
	}
}
