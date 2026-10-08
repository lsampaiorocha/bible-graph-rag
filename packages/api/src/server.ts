import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { serve } from "@hono/node-server";
import { runAgent, runClosedBook, runLocal, runVector } from "@kgrag/retrieval";
import { Hono } from "hono";
import { z } from "zod";
import { embedQuery } from "./embedder.ts";
import { llm, source } from "./setup.ts";

const PORT = 8787;
const LATEST_QA = fileURLToPath(
	new URL("../../../eval/results/qa_latest.json", import.meta.url),
);
const LATEST_REPORT = fileURLToPath(
	new URL("../../../eval/results/latest.json", import.meta.url),
);
const QueryRequest = z.object({
	text: z.string().min(1).max(500),
	mode: z.enum(["local", "agent", "vector", "closed_book"]),
	// false = keyword planning, no LLM: what the public site does without a key.
	llm: z.boolean().default(true),
});

const app = new Hono();
app.get("/api/eval", (context) =>
	existsSync(LATEST_REPORT)
		? context.json(JSON.parse(readFileSync(LATEST_REPORT, "utf-8")))
		: context.json(
				{ error: "no report yet; run uv run kgrag-eval report --index gen-v1" },
				404,
			),
);
app.get("/api/eval/qa", (context) =>
	existsSync(LATEST_QA)
		? context.json(JSON.parse(readFileSync(LATEST_QA, "utf-8")))
		: context.json({ error: "no QA run scored yet" }, 404),
);
app.get("/api/graph", async (context) =>
	context.json(await source.graphView()),
);
app.get("/api/entity/:id", async (context) => {
	const details = await source.entityDetails(context.req.param("id"));
	return details
		? context.json(details)
		: context.json({ error: "unknown entity" }, 404);
});
app.post("/api/query", async (context) => {
	const request = QueryRequest.safeParse(
		await context.req.json().catch(() => null),
	);
	if (!request.success)
		return context.json({ error: z.prettifyError(request.error) }, 400);
	const { text, mode } = request.data;
	if (mode === "local") return context.json(await runLocal(source, text));
	if (mode === "vector")
		return context.json(await runVector(source, text, llm, embedQuery));
	if (mode === "closed_book")
		return context.json(await runClosedBook(source, text, llm));
	return context.json(
		await runAgent(source, text, request.data.llm ? llm : null, embedQuery),
	);
});

serve({ fetch: app.fetch, port: PORT }, () =>
	console.log(`API listening on http://localhost:${PORT}`),
);

// Load the embedding model at startup so the first question does not wait for it.
void embedQuery("warm up").catch((error) =>
	console.error("embedding model failed to load:", error),
);
