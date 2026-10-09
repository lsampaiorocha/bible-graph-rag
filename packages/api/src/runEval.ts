import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import {
	type RetrievalTrace,
	runAgent,
	runClosedBook,
	runVector,
} from "@kgrag/retrieval";
import { parse } from "yaml";
import { embedQuery } from "./embedder.ts";
import { driver, llm, source } from "./setup.ts";

const QUESTIONS = fileURLToPath(
	new URL("../../../eval/questions.yaml", import.meta.url),
);
const RUNS_DIR = fileURLToPath(new URL("../../../eval/runs/", import.meta.url));
const MODES = {
	agent: (question: string) => runAgent(source, question, llm, embedQuery),
	vector: (question: string) => runVector(source, question, llm, embedQuery),
	closed_book: (question: string) => runClosedBook(source, question, llm),
} satisfies Record<string, (question: string) => Promise<RetrievalTrace>>;
const CONCURRENCY = 3;

interface Question {
	id: string;
	question: string;
}

// --run adds to an existing run folder; --only and --modes limit it to some question ids
// and modes.
const { values: options } = parseArgs({
	options: {
		run: { type: "string" },
		only: { type: "string" },
		modes: { type: "string" },
	},
});
const only = options.only?.split(",");
const questions = (
	parse(readFileSync(QUESTIONS, "utf-8")) as Question[]
).filter((question) => !only || only.includes(question.id));
const runId =
	options.run ?? new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
const jobs = questions.flatMap((question) =>
	Object.entries(MODES)
		.filter(
			([mode]) => !options.modes || options.modes.split(",").includes(mode),
		)
		.map(([mode, run]) => ({ mode, question, run })),
);
let done = 0;
let cost = 0;

async function worker(): Promise<void> {
	for (let job = jobs.shift(); job; job = jobs.shift()) {
		const trace = await job.run(job.question.question);
		const dir = `${RUNS_DIR}${runId}/${job.mode}`;
		mkdirSync(dir, { recursive: true });
		writeFileSync(
			`${dir}/${job.question.id}.json`,
			JSON.stringify(trace, null, 1),
		);
		cost += trace.metrics.usage?.cost_usd ?? 0;
		done += 1;
		console.log(
			`[${done}] ${job.mode.padEnd(11)} ${job.question.id} ${trace.warnings ? "WARN " : ""}$${cost.toFixed(4)}`,
		);
	}
}

await Promise.all(Array.from({ length: CONCURRENCY }, worker));
await driver.close();
console.log(
	`run ${runId}: ${done} traces in eval/runs/${runId}, generation cost $${cost.toFixed(4)}`,
);
