import { serve } from "@hono/node-server";
import { runLocal } from "@kgrag/retrieval";
import { Hono } from "hono";
import neo4j from "neo4j-driver";
import { z } from "zod";
import { Neo4jGraphSource } from "./neo4jGraphSource.ts";

const PORT = 8787;
const QueryRequest = z.object({
	text: z.string().min(1).max(500),
	mode: z.literal("local"),
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

const app = new Hono();
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
	return context.json(await runLocal(source, request.data.text));
});

serve({ fetch: app.fetch, port: PORT }, () =>
	console.log(`API listening on http://localhost:${PORT}`),
);
