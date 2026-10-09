import { z } from "zod";

export const Status = z.enum(["candidate", "context", "cited"]);
export const Mode = z.enum([
	"local",
	"path",
	"global",
	"global_lite",
	"vector",
	"closed_book",
	"agent",
]);
const PrunedReason = z.enum(["neighbour_cap", "rank_cutoff", "token_budget"]);

export const Seed = z.object({
	node_id: z.string(),
	method: z.enum(["alias_exact", "alias_fuzzy", "embedding", "vector_chunk"]),
	matched_text: z.string().optional(),
	score: z.number(),
});

export const TraceNode = z.object({
	id: z.string(),
	label: z.string().optional(),
	hop: z.number().int(),
	score: z.number(),
	status: Status,
	handle: z.string().optional(),
	reached_via: z.string().optional(),
	pruned_reason: PrunedReason.optional(),
});

export const TraceEdge = z.object({
	id: z.string(),
	source: z.string(),
	target: z.string(),
	type: z.string(),
	hop: z.number().int(),
	score: z.number(),
	status: Status,
	handle: z.string().optional(),
	evidence_chunk_ids: z.array(z.string()),
	pruned_reason: PrunedReason.optional(),
});

export const TracePath = z.object({
	rank: z.number().int(),
	cost: z.number(),
	status: Status,
	node_ids: z.array(z.string()),
	edge_ids: z.array(z.string()),
});

export const TraceCommunity = z.object({
	id: z.string(),
	level: z.number().int(),
	score: z.number(),
	status: Status,
	handle: z.string().optional(),
	method: z.enum(["contains_seed", "summary_embedding", "map_reduce"]),
	map_helpfulness: z.number().optional(),
});

export const TraceChunk = z.object({
	id: z.string(),
	rank: z.number().int(),
	score: z.number(),
	status: Status,
	handle: z.string().optional(),
	retrieved_by: z.array(z.string()),
	highlight_entity_ids: z.array(z.string()),
});

export const TraceStep = z.object({
	index: z.number().int(),
	tool: z.enum([
		"find_entity",
		"explore",
		"ancestors",
		"descendants",
		"relatives",
		"path",
	]),
	args: z.record(z.string(), z.union([z.string(), z.number()])),
	entity_id: z.string().optional(),
	candidates: z.array(z.string()).optional(),
	node_ids: z.array(z.string()),
	edge_ids: z.array(z.string()),
	note: z.string().optional(),
	ms: z.number(),
});

export const Answer = z.object({
	text: z.string(),
	sentences: z.array(
		z.object({ text: z.string(), cites: z.array(z.string()) }),
	),
	invalid_handles: z.array(z.string()),
	uncited_sentence_idx: z.array(z.number().int()),
	insufficient_context: z.boolean(),
});

export const RetrievalTrace = z.object({
	trace_version: z.literal("1.0"),
	trace_id: z.string(),
	index_id: z.string(),
	created_at: z.string(),
	source: z.enum(["live", "precomputed"]),
	query: z.object({
		text: z.string(),
		mode: Mode,
		params: z.record(z.string(), z.union([z.number(), z.string()])),
	}),
	seeds: z.array(Seed),
	steps: z.array(TraceStep).optional(),
	nodes: z.array(TraceNode),
	edges: z.array(TraceEdge),
	paths: z.array(TracePath),
	communities: z.array(TraceCommunity),
	chunks: z.array(TraceChunk),
	context: z.object({
		handles: z.record(z.string(), z.string()),
		token_count: z.number().int(),
		token_budget: z.number().int(),
		truncated: z.boolean(),
	}),
	answer: Answer.nullable(),
	warnings: z.array(z.string()).optional(),
	metrics: z.object({
		timings_ms: z.record(z.string(), z.number()),
		usage: z
			.object({
				model: z.string(),
				input_tokens: z.number().int(),
				output_tokens: z.number().int(),
				cost_usd: z.number(),
			})
			.optional(),
	}),
});

export type Status = z.infer<typeof Status>;
export type Seed = z.infer<typeof Seed>;
export type TraceNode = z.infer<typeof TraceNode>;
export type TraceEdge = z.infer<typeof TraceEdge>;
export type TraceStep = z.infer<typeof TraceStep>;
export type TraceChunk = z.infer<typeof TraceChunk>;
export type Answer = z.infer<typeof Answer>;
export type TracePath = z.infer<typeof TracePath>;
export type RetrievalTrace = z.infer<typeof RetrievalTrace>;
