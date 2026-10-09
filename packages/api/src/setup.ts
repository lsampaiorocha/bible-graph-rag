import type { LlmClient } from "@kgrag/retrieval";
import neo4j from "neo4j-driver";
import { ClaudeCliClient } from "./claudeCli.ts";
import { Neo4jGraphSource } from "./neo4jGraphSource.ts";
import { OpenAiCompatibleClient } from "./openAiClient.ts";

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
const projectBudgetUsd = Number(process.env.KGRAG_PROJECT_BUDGET_USD ?? 40);
const provider = process.env.KGRAG_LLM_PROVIDER ?? "claude-code";
if (provider !== "claude-code" && provider !== "openai")
	throw new Error(`KGRAG_LLM_PROVIDER=${provider}; use claude-code or openai`);
export const llm: LlmClient =
	provider === "openai"
		? new OpenAiCompatibleClient({
				baseUrl: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
				apiKey: process.env.OPENAI_API_KEY ?? "none",
				model: process.env.KGRAG_AGENT_MODEL ?? "gpt-4.1-mini",
				maxCallUsd: 0.1,
				projectBudgetUsd,
				priceInputPerMtok: Number(process.env.KGRAG_PRICE_INPUT_PER_MTOK ?? 0),
				priceOutputPerMtok: Number(
					process.env.KGRAG_PRICE_OUTPUT_PER_MTOK ?? 0,
				),
			})
		: new ClaudeCliClient({
				model: process.env.KGRAG_AGENT_MODEL ?? "claude-haiku-5-5",
				effort: process.env.KGRAG_AGENT_EFFORT ?? "medium",
				maxCallUsd: 0.1,
				projectBudgetUsd,
			});
