import { spawn } from "node:child_process";
import type { LlmClient, LlmUsage } from "@kgrag/retrieval";
import { appendLedger, ledgerTotal } from "./ledger.ts";

const CALL_TIMEOUT_MS = 180_000;

export interface ClaudeCliOptions {
	model: string;
	effort: string;
	maxCallUsd: number;
	projectBudgetUsd: number;
}

interface ClaudeResponse {
	is_error?: boolean;
	result?: string;
	structured_output?: unknown;
	total_cost_usd?: number;
	duration_ms?: number;
	usage?: {
		input_tokens?: number;
		cache_creation_input_tokens?: number;
		cache_read_input_tokens?: number;
		output_tokens?: number;
		output_tokens_details?: { thinking_tokens?: number };
	};
}

function runClaude(args: string[], input: string): Promise<ClaudeResponse> {
	return new Promise((resolve, reject) => {
		const child = spawn("claude", args, { stdio: ["pipe", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		const timer = setTimeout(() => {
			child.kill();
			reject(
				new Error(`claude -p timed out after ${CALL_TIMEOUT_MS / 1000} s`),
			);
		}, CALL_TIMEOUT_MS);
		child.stdout.on("data", (chunk) => {
			stdout += chunk;
		});
		child.stderr.on("data", (chunk) => {
			stderr += chunk;
		});
		child.on("error", reject);
		child.on("close", () => {
			clearTimeout(timer);
			try {
				resolve(JSON.parse(stdout));
			} catch {
				reject(
					new Error(`claude -p returned no JSON: ${stderr.slice(0, 300)}`),
				);
			}
		});
		child.stdin.end(input);
	});
}

// The same isolated `claude -p` call the Python pipeline makes, logged to the same
// ledger and held to the same project budget.
export class ClaudeCliClient implements LlmClient {
	private readonly options: ClaudeCliOptions;

	constructor(options: ClaudeCliOptions) {
		this.options = options;
	}

	async complete<T>(request: {
		purpose: string;
		system: string;
		prompt: string;
		schema: Record<string, unknown>;
	}): Promise<{ output: T; usage: LlmUsage }> {
		const { model, effort, maxCallUsd, projectBudgetUsd } = this.options;
		const spent = ledgerTotal();
		if (spent + maxCallUsd > projectBudgetUsd) {
			throw new Error(
				`project budget reached: $${spent.toFixed(2)} of $${projectBudgetUsd.toFixed(2)}`,
			);
		}
		const response = await runClaude(
			[
				"-p",
				"--setting-sources",
				"",
				"--strict-mcp-config",
				"--tools",
				"",
				"--no-session-persistence",
				"--model",
				model,
				"--effort",
				effort,
				"--output-format",
				"json",
				"--json-schema",
				JSON.stringify(request.schema),
				"--system-prompt",
				request.system,
				"--max-budget-usd",
				maxCallUsd.toFixed(4),
			],
			request.prompt,
		);
		const usage = response.usage ?? {};
		const inputTokens =
			(usage.input_tokens ?? 0) +
			(usage.cache_creation_input_tokens ?? 0) +
			(usage.cache_read_input_tokens ?? 0);
		appendLedger({
			time: `${new Date().toISOString().slice(0, 19)}+00:00`,
			run_id: `api-${new Date().toISOString().slice(0, 10)}`,
			purpose: request.purpose,
			model,
			effort,
			item_id: request.prompt.slice(0, 80),
			attempt: 1,
			is_error: Boolean(response.is_error),
			input_tokens: usage.input_tokens ?? 0,
			cache_creation_input_tokens: usage.cache_creation_input_tokens ?? 0,
			cache_read_input_tokens: usage.cache_read_input_tokens ?? 0,
			output_tokens: usage.output_tokens ?? 0,
			thinking_tokens: usage.output_tokens_details?.thinking_tokens ?? 0,
			cost_usd: response.total_cost_usd ?? 0,
			duration_ms: response.duration_ms ?? 0,
		});
		if (response.is_error)
			throw new Error(String(response.result).slice(0, 300));
		if (response.structured_output === undefined)
			throw new Error("claude -p returned no structured output");
		return {
			output: response.structured_output as T,
			usage: {
				model,
				input_tokens: inputTokens,
				output_tokens: usage.output_tokens ?? 0,
				cost_usd: response.total_cost_usd ?? 0,
			},
		};
	}
}
