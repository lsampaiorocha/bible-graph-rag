import { ANSWER_SCHEMA, ANSWER_SYSTEM, type LlmClient } from "../agent.ts";
import { applyAnswer, type RawAnswer } from "../context.ts";
import type { Embedder, GraphSource } from "../graphSource.ts";
import type { RetrievalTrace, TraceChunk } from "../trace.ts";

export const VECTOR_TOP_K = 5;

export const CLOSED_BOOK_SYSTEM = `You answer a question about the King James Bible from memory; no documents are given. Answer in a few plain sentences, naming the people involved. When the question asks for all of someone's ancestors or descendants, name every one you know, one sentence per parent. Leave every cites list empty. If you do not know, say so and set insufficient_context to true.`;

function baseTrace(
	question: string,
	mode: "vector" | "closed_book",
	indexId: string,
): RetrievalTrace {
	return {
		trace_version: "1.0",
		trace_id: crypto.randomUUID(),
		index_id: indexId,
		created_at: new Date().toISOString(),
		source: "live",
		query: {
			text: question,
			mode,
			params: mode === "vector" ? { top_k: VECTOR_TOP_K } : {},
		},
		seeds: [],
		nodes: [],
		edges: [],
		paths: [],
		communities: [],
		chunks: [],
		context: { handles: {}, token_count: 0, token_budget: 0, truncated: false },
		answer: null,
		metrics: { timings_ms: {} },
	};
}

// The plain-RAG baseline: the passages nearest the question, then the same answer prompt
// and citation checks the graph agent uses, so the comparison isolates retrieval.
export async function runVector(
	source: GraphSource,
	question: string,
	llm: LlmClient,
	embed: Embedder,
): Promise<RetrievalTrace> {
	const trace = baseTrace(question, "vector", await source.indexId());
	const retrieveStarted = performance.now();
	const passages = await source.passagesByVector(
		await embed(question),
		VECTOR_TOP_K,
	);
	trace.metrics.timings_ms.retrieve = Math.round(
		performance.now() - retrieveStarted,
	);
	const handles: Record<string, string> = {};
	const chunks: TraceChunk[] = passages.map((passage, index) => {
		const handle = `C${index + 1}`;
		handles[handle] = passage.id;
		return {
			id: passage.id,
			rank: index + 1,
			score: passage.score,
			status: "context",
			handle,
			retrieved_by: ["vector"],
			highlight_entity_ids: [],
		};
	});
	// The passage text carries [Gen.x.y] verse markers and the model often cites those;
	// they point into a passage it was given, so they count as citing that passage.
	for (const passage of passages)
		for (const ref of passage.verseRefs) handles[ref] = passage.id;
	const context = passages
		.map((passage, index) => `[C${index + 1}] ${passage.id}\n${passage.text}`)
		.join("\n\n");
	trace.chunks = chunks;
	trace.context = {
		handles,
		token_count: Math.round(context.length / 4),
		token_budget: 0,
		truncated: false,
	};
	const generateStarted = performance.now();
	try {
		const result = await llm.complete<RawAnswer>({
			purpose: "vector_answer",
			system: ANSWER_SYSTEM,
			prompt: `Question: ${question}\n\n## Passages\n${context}`,
			schema: ANSWER_SCHEMA,
		});
		trace.answer = applyAnswer(result.output, handles, [], [], chunks);
		trace.metrics.usage = result.usage;
	} catch (error) {
		trace.warnings = [`answer failed: ${String(error).slice(0, 200)}`];
	}
	trace.metrics.timings_ms.generate = Math.round(
		performance.now() - generateStarted,
	);
	return trace;
}

// The no-retrieval baseline: what the model says from memory alone.
export async function runClosedBook(
	source: GraphSource,
	question: string,
	llm: LlmClient,
): Promise<RetrievalTrace> {
	const trace = baseTrace(question, "closed_book", await source.indexId());
	const started = performance.now();
	try {
		const result = await llm.complete<RawAnswer>({
			purpose: "closed_book_answer",
			system: CLOSED_BOOK_SYSTEM,
			prompt: `Question: ${question}`,
			schema: ANSWER_SCHEMA,
		});
		trace.answer = applyAnswer(result.output, {}, [], [], []);
		trace.metrics.usage = result.usage;
	} catch (error) {
		trace.warnings = [`answer failed: ${String(error).slice(0, 200)}`];
	}
	trace.metrics.timings_ms.generate = Math.round(performance.now() - started);
	return trace;
}
