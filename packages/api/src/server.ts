import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { serve } from "@hono/node-server";
import { runAgent, runLocal } from "@kgrag/retrieval";
import { Hono } from "hono";
import neo4j from "neo4j-driver";
import { z } from "zod";
import { ClaudeCliClient } from "./claudeCli.ts";
import { embedQuery } from "./embedder.ts";
import { Neo4jGraphSource } from "./neo4jGraphSource.ts";

const PORT = 8787;
const LATEST_REPORT = fileURLToPath(
	new URL("../../../eval/results/latest.json", import.meta.url),
);
const QueryRequest = z.object({
	text: z.string().min(1).max(500),
	mode: z.enum(["local", "agent"]),
	// false = keyword planning, no LLM: what the public site does without a key.
	llm: z.boolean().default(true),
});

process.loadEnvFile(new URL("../../../.env", import.meta.url));
const password = process.env.NEO4J_PASSWORD;
if (!password) throw new Error("NEO4J_PASSWORD is not set in .env");

const driver = neo4j.driver(
	process.env.NEO4J_URI ?? "neo4j://localhost:7687",
	neo4j.auth.basic("neo4j", password),
	{
		disableLosslessIntegers: true,
	},
);
const source = new Neo4jGraphSource(driver);
const llm = new ClaudeCliClient({
	model: process.env.KGRAG_AGENT_MODEL ?? "claude-haiku-5-5",
	effort: process.env.KGRAG_AGENT_EFFORT ?? "medium",
	maxCallUsd: 0.1,
	projectBudgetUsd: Number(process.env.KGRAG_PROJECT_BUDGET_USD ?? 40),
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
