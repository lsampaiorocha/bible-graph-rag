import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

// The ledger the Python pipeline writes, so `kgrag usage` and the project budget see
// the agent's spend too.
export const LEDGER_PATH = fileURLToPath(
	new URL("../../../data/ledger/llm_calls.jsonl", import.meta.url),
);

export function ledgerTotal(path = LEDGER_PATH): number {
	if (!existsSync(path)) return 0;
	return readFileSync(path, "utf-8")
		.split("\n")
		.filter(Boolean)
		.reduce((sum, line) => sum + (JSON.parse(line).cost_usd ?? 0), 0);
}

export function appendLedger(
	row: Record<string, unknown>,
	path = LEDGER_PATH,
): void {
	mkdirSync(dirname(path), { recursive: true });
	appendFileSync(path, `${JSON.stringify(row)}\n`);
}
