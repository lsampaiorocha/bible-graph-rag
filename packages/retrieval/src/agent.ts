import { applyAnswer, buildContext, type RawAnswer } from "./context.ts";
import {
	compareText,
	type Embedder,
	type GraphSource,
	MAX_LINEAGE_DEPTH,
	RELATIONS,
} from "./graphSource.ts";
import { findSeeds, type SeedMatch, type Subgraph } from "./modes/local.ts";
import {
	executeCall,
	MAX_TOOL_CALLS,
	MEANING_MATCH_MIN,
	OVERVIEW_WORDS,
	planByKeywords,
	TOOLS,
	type ToolCall,
	type ToolContext,
} from "./tools.ts";
import type { RetrievalTrace, TraceStep } from "./trace.ts";

export interface LlmUsage {
	model: string;
	input_tokens: number;
	output_tokens: number;
	cost_usd: number;
}

export interface LlmClient {
	complete<T>(request: {
		purpose: string;
		system: string;
		prompt: string;
		schema: Record<string, unknown>;
	}): Promise<{ output: T; usage: LlmUsage }>;
}

interface Plan {
	calls: ToolCall[];
	reasoning: string;
}

const nullable = (schema: Record<string, unknown>) => ({
	anyOf: [schema, { type: "null" }],
});

export const PLAN_SCHEMA = {
	type: "object",
	properties: {
		reasoning: { type: "string" },
		calls: {
			type: "array",
			maxItems: MAX_TOOL_CALLS,
			items: {
				type: "object",
				properties: {
					tool: { enum: [...TOOLS] },
					entity: { type: "string" },
					relation: nullable({ enum: [...RELATIONS] }),
					generations: nullable({
						type: "integer",
						minimum: 1,
						maximum: MAX_LINEAGE_DEPTH,
					}),
					other: nullable({ type: "string" }),
					family_only: nullable({ type: "boolean" }),
				},
				required: [
					"tool",
					"entity",
					"relation",
					"generations",
					"other",
					"family_only",
				],
				additionalProperties: false,
			},
		},
	},
	required: ["reasoning", "calls"],
	additionalProperties: false,
};

export const ANSWER_SCHEMA = {
	type: "object",
	properties: {
		sentences: {
			type: "array",
			items: {
				type: "object",
				properties: {
					text: { type: "string" },
					cites: { type: "array", items: { type: "string" } },
				},
				required: ["text", "cites"],
				additionalProperties: false,
			},
		},
		insufficient_context: { type: "boolean" },
	},
	required: ["sentences", "insufficient_context"],
	additionalProperties: false,
};

export const PLANNER_SYSTEM = `You plan how to answer a question about the King James Bible from a knowledge graph of its people, places, groups and events. You do not answer; you choose tool calls, at most ${MAX_TOOL_CALLS} in total.

Tools (entity = an entity id from the list you are given, or a name):
- ancestors(entity, generations): a person's parents, grandparents and so on, up to ${MAX_LINEAGE_DEPTH} generations. Use for ancestors, lineage, genealogy, "descended from".
- descendants(entity, generations): a person's children, grandchildren and so on.
- relatives(entity, relation): one kind of direct family tie; relation is one of ${RELATIONS.join(", ")}.
- explore(entity): everything closely linked to an entity: people, places, events. Use for open questions such as what happened, who someone was, why something happened.
- path(entity, other, family_only): the shortest chains of links joining two entities, avoiding hubs such as God. With family_only true it follows only parent, sibling and spouse links, up to 6; otherwise any links, up to 4. Use for how one person or place is related or connected to another; family_only for kinship. Always set both entity and other.
- overview(entity): the whole book at a glance; entity = people for its main characters and the links among them, or events for the leading event of each of its main stories, in story order. Use for questions about the main characters, the most important events or what happens in the book overall.
- find_entity(entity): list every entity with that name. Use when the question could mean several people with the same name and the list does not settle it.

Rules:
- Prefer the most specific tool: a family question goes to ancestors, descendants or relatives, not explore.
- When several entities share a name, pick the id whose description fits the question.
- Set relation only for relatives, generations only for ancestors or descendants, and other and family_only only for path; otherwise null.
- Leave generations null to get every generation; set a number only when the question limits it (children = 1, grandchildren = 2).
- If a previous round is shown, use it to fix what failed (an ambiguous name, nothing found); do not repeat a call that worked.`;

export const ANSWER_SYSTEM = `You answer a question about the King James Bible using only the context given: entities [E…], relationships [R…] and verses [C…]. Do not add facts from memory.

- Answer in a few plain sentences. For a family line, name the people in order.
- For an overview of a book's main characters or main events, give one sentence for each, in story order, saying why it matters.
- When the question asks for all of someone's ancestors or descendants, name every one the context gives, not a sample: one sentence per parent, such as "Jacob's sons by Leah: Reuben, Simeon, Levi, Judah, Issachar and Zebulun."
- After each sentence, cite the handles that support it, such as ["R1", "C2"]. Every sentence needs at least one citation; prefer verses [C…] and relationships [R…].
- If the context does not answer the question, say what it does show and set insufficient_context to true.`;

async function seedCards(
	source: GraphSource,
	seeds: SeedMatch[],
): Promise<string> {
	const cards = await source.entityCards(seeds.map(({ seed }) => seed.node_id));
	if (cards.length === 0)
		return "No entity names were recognized in the question.";
	return cards
		.map(
			(card) =>
				`- ${card.id}: ${card.label} (${card.kind}${card.firstVerse ? `, first named ${card.firstVerse}` : ""}): ${card.description}`,
		)
		.join("\n");
}

function observations(steps: TraceStep[]): string {
	return steps
		.map(
			(step) =>
				`- ${step.tool}(${JSON.stringify(step.args)}) → ${step.entity_id ?? "unresolved"}: ` +
				`${step.edge_ids.length} links, ${step.node_ids.length} entities` +
				(step.note ? `; ${step.note}` : "") +
				(step.candidates?.length
					? `; candidates ${step.candidates.join(", ")}`
					: ""),
		)
		.join("\n");
}

export async function runAgent(
	source: GraphSource,
	question: string,
	llm: LlmClient | null,
	embed?: Embedder,
): Promise<RetrievalTrace> {
	const timings: Record<string, number> = {};
	const warnings: string[] = [];
	const usage: LlmUsage[] = [];
	let questionVector: number[] | null = null;
	if (embed) {
		try {
			questionVector = await embed(question);
		} catch (error) {
			warnings.push(
				`question embedding failed: ${String(error).slice(0, 200)}`,
			);
		}
	}
	const seeds = await findSeeds(source, question);
	// A question that names nobody ("Who built the ark?") starts from the entities
	// nearest to it in meaning, except a whole-book question, where that only lures the
	// planner away ("the events of Genesis" lands near Noah).
	if (
		seeds.length === 0 &&
		questionVector &&
		!OVERVIEW_WORDS.test(question.toLowerCase())
	) {
		for (const hit of await source.seedsByVector(questionVector, 3)) {
			if (hit.score >= MEANING_MATCH_MIN) {
				seeds.push({
					seed: { node_id: hit.nodeId, method: "embedding", score: hit.score },
					label: hit.label,
				});
			}
		}
	}
	const toolContext: ToolContext = { embed, questionVector };
	const known = await seedCards(source, seeds);

	const plan = async (
		previous: TraceStep[],
		remaining: number,
	): Promise<ToolCall[]> => {
		if (!llm) return previous.length ? [] : planByKeywords(question, seeds);
		try {
			const prompt =
				`Question: ${question}\n\nEntities named in the question:\n${known}` +
				(previous.length
					? `\n\nPrevious round:\n${observations(previous)}\n\nCalls left: ${remaining}`
					: "");
			const result = await llm.complete<Plan>({
				purpose: "agent_plan",
				system: PLANNER_SYSTEM,
				prompt,
				schema: PLAN_SCHEMA,
			});
			usage.push(result.usage);
			return result.output.calls.slice(0, remaining);
		} catch (error) {
			warnings.push(
				`planner failed, used keyword rules: ${String(error).slice(0, 200)}`,
			);
			return previous.length ? [] : planByKeywords(question, seeds);
		}
	};

	const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };
	const steps: TraceStep[] = [];
	for (let round = 0; round < 2 && steps.length < MAX_TOOL_CALLS; round++) {
		// A second round runs only when the first left something unresolved.
		if (round === 1 && !steps.some((step) => step.note)) break;
		const roundStarted = performance.now();
		const calls = await plan(steps, MAX_TOOL_CALLS - steps.length);
		timings[`plan_${round + 1}`] = Math.round(performance.now() - roundStarted);
		for (const planned of calls) {
			if (steps.length >= MAX_TOOL_CALLS) break;
			// The planner often leaves other empty on path even with both entities listed;
			// the other entity named in the question is the one it means.
			const call =
				planned.tool === "path" && !planned.other
					? {
							...planned,
							other:
								seeds
									.map(({ seed }) => seed.node_id)
									.find((id) => id !== planned.entity) ?? null,
						}
					: planned;
			steps.push(
				await executeCall(
					source,
					call,
					steps.length + 1,
					subgraph,
					toolContext,
				),
			);
		}
		if (calls.length === 0) break;
	}
	if (steps.length === 0) {
		for (const call of planByKeywords(question, seeds))
			steps.push(
				await executeCall(
					source,
					call,
					steps.length + 1,
					subgraph,
					toolContext,
				),
			);
	}
	timings.retrieve = steps.reduce((sum, step) => sum + step.ms, 0);

	const nodes = [...subgraph.nodes.values()].sort(
		(a, b) => a.hop - b.hop || compareText(a.id, b.id),
	);
	const edges = [...subgraph.edges.values()].sort(
		(a, b) => a.hop - b.hop || compareText(a.id, b.id),
	);
	const contextStarted = performance.now();
	const context = await buildContext(source, nodes, edges);
	timings.assemble = Math.round(performance.now() - contextStarted);

	let answer = null;
	if (llm && Object.keys(context.handles).length > 0) {
		const answerStarted = performance.now();
		try {
			const result = await llm.complete<RawAnswer>({
				purpose: "agent_answer",
				system: ANSWER_SYSTEM,
				prompt: `Question: ${question}\n\n${context.text}`,
				schema: ANSWER_SCHEMA,
			});
			usage.push(result.usage);
			answer = applyAnswer(
				result.output,
				context.handles,
				nodes,
				edges,
				context.chunks,
			);
		} catch (error) {
			warnings.push(`answer failed: ${String(error).slice(0, 200)}`);
		}
		timings.generate = Math.round(performance.now() - answerStarted);
	}

	const [firstUsage] = usage;
	return {
		trace_version: "1.0",
		trace_id: crypto.randomUUID(),
		index_id: await source.indexId(),
		created_at: new Date().toISOString(),
		source: "live",
		query: {
			text: question,
			mode: "agent",
			params: {
				max_tool_calls: MAX_TOOL_CALLS,
				planner: llm ? "llm" : "keywords",
			},
		},
		seeds: seeds.map(({ seed }) => seed),
		steps,
		nodes,
		edges,
		paths: (subgraph.paths ?? []).map((path) => ({
			...path,
			status: path.edge_ids.every(
				(id) => edges.find((edge) => edge.id === id)?.status === "cited",
			)
				? ("cited" as const)
				: path.status,
		})),
		communities: [],
		chunks: context.chunks,
		context: {
			handles: context.handles,
			token_count: context.tokenCount,
			token_budget: 0,
			truncated: context.truncated,
		},
		answer,
		...(warnings.length ? { warnings } : {}),
		metrics: {
			timings_ms: timings,
			...(firstUsage
				? {
						usage: {
							model: firstUsage.model,
							input_tokens: usage.reduce(
								(sum, item) => sum + item.input_tokens,
								0,
							),
							output_tokens: usage.reduce(
								(sum, item) => sum + item.output_tokens,
								0,
							),
							cost_usd: usage.reduce((sum, item) => sum + item.cost_usd, 0),
						},
					}
				: {}),
		},
	};
}
