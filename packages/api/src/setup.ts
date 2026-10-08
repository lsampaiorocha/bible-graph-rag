import neo4j from "neo4j-driver";
import { ClaudeCliClient } from "./claudeCli.ts";
import { Neo4jGraphSource } from "./neo4jGraphSource.ts";

process.loadEnvFile(new URL("../../../.env", import.meta.url));
const password = process.env.NEO4J_PASSWORD;
if (!password) throw new Error("NEO4J_PASSWORD is not set in .env");

export const driver = neo4j.driver(
	process.env.NEO4J_URI ?? "neo4j://localhost:7687",
	neo4j.auth.basic("neo4j", password),
	{
		disableLosslessIntegers: true,
	},
);
export const source = new Neo4jGraphSource(driver);
export const llm = new ClaudeCliClient({
	model: process.env.KGRAG_AGENT_MODEL ?? "claude-haiku-5-5",
	effort: process.env.KGRAG_AGENT_EFFORT ?? "medium",
	maxCallUsd: 0.1,
	projectBudgetUsd: Number(process.env.KGRAG_PROJECT_BUDGET_USD ?? 40),
});
