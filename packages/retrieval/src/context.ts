import {
	compareText,
	type GraphSource,
	MAX_LINEAGE_LINKS,
} from "./graphSource.ts";
import type { Answer, TraceChunk, TraceEdge, TraceNode } from "./trace.ts";

// Room for a whole family tree, which is the answer to a lineage question;
// each explore result is already cut to its top 25 before it gets here.
export const MAX_CONTEXT_NODES = 150;
export const MAX_CONTEXT_EDGES = MAX_LINEAGE_LINKS;
export const MAX_CONTEXT_VERSES = 80;

export interface AnswerContext {
	text: string;
	handles: Record<string, string>;
	chunks: TraceChunk[];
	tokenCount: number;
	truncated: boolean;
}

export interface RawAnswer {
	sentences: { text: string; cites: string[] }[];
	insufficient_context: boolean;
}

function keep<T extends TraceNode | TraceEdge>(
	elements: T[],
	limit: number,
): { kept: T[]; truncated: boolean } {
	const candidates = elements
		.filter((element) => element.status === "context")
		.sort((a, b) => a.hop - b.hop || compareText(a.id, b.id));
	for (const dropped of candidates.slice(limit)) {
		dropped.status = "candidate";
		dropped.pruned_reason = "token_budget";
		delete dropped.handle;
	}
	return {
		kept: candidates.slice(0, limit),
		truncated: candidates.length > limit,
	};
}

// Assigns the short handles the LLM cites (E entities, R relationships, C verses) and
// writes the context it answers from. Mutates node and edge handles in place.
export async function buildContext(
	source: GraphSource,
	nodes: TraceNode[],
	edges: TraceEdge[],
): Promise<AnswerContext> {
	const keptNodes = keep(nodes, MAX_CONTEXT_NODES);
	const keptEdges = keep(edges, MAX_CONTEXT_EDGES);
	const cards = new Map(
		(await source.entityCards(keptNodes.kept.map((node) => node.id))).map(
			(c) => [c.id, c],
		),
	);
	const details = new Map(
		(await source.edgeDetails(keptEdges.kept.map((edge) => edge.id))).map(
			(d) => [d.id, d],
		),
	);
	const labelOf = new Map(
		nodes.map((node) => [node.id, node.label ?? node.id]),
	);

	const verseRefs = [
		...new Set(
			keptEdges.kept.flatMap(
				(edge) => details.get(edge.id)?.evidenceVerses ?? [],
			),
		),
	];
	const verseText = new Map(
		(await source.verses(verseRefs.slice(0, MAX_CONTEXT_VERSES))).map(
			(verse) => [verse.id, verse.text],
		),
	);

	const handles: Record<string, string> = {};
	const lines = ["## Entities"];
	keptNodes.kept.forEach((node, index) => {
		node.handle = `E${index + 1}`;
		handles[node.handle] = node.id;
		const card = cards.get(node.id);
		const facts = [
			card?.kind,
			card?.gender && card.gender !== "unknown" ? card.gender : null,
		].filter(Boolean);
		const first = card?.firstVerse ? ` First named ${card.firstVerse}.` : "";
		lines.push(
			`[${node.handle}] ${labelOf.get(node.id)} (${facts.join(", ")}): ${card?.description ?? ""}${first}`,
		);
	});
	lines.push("", "## Relationships");
	keptEdges.kept.forEach((edge, index) => {
		edge.handle = `R${index + 1}`;
		handles[edge.handle] = edge.id;
		const detail = details.get(edge.id);
		const type = detail?.otherLabel
			? `${edge.type} (${detail.otherLabel})`
			: edge.type;
		const verses = detail?.evidenceVerses.length
			? ` (${detail.evidenceVerses.join(", ")})`
			: "";
		lines.push(
			`[${edge.handle}] ${labelOf.get(edge.source)} —${type}→ ${labelOf.get(edge.target)}: ${detail?.description ?? ""}${verses}`,
		);
		edge.evidence_chunk_ids = (detail?.evidenceVerses ?? []).filter((ref) =>
			verseText.has(ref),
		);
	});

	lines.push("", "## Verses");
	const chunks: TraceChunk[] = [...verseText.keys()].map((ref, index) => {
		const handle = `C${index + 1}`;
		handles[handle] = ref;
		lines.push(`[${handle}] ${ref}: ${verseText.get(ref)}`);
		const supporting = keptEdges.kept.filter((edge) =>
			edge.evidence_chunk_ids.includes(ref),
		);
		return {
			id: ref,
			rank: index + 1,
			score: 1,
			status: "context",
			handle,
			retrieved_by: supporting.map((edge) => `evidence:${edge.id}`),
			highlight_entity_ids: [
				...new Set(supporting.flatMap((edge) => [edge.source, edge.target])),
			],
		};
	});

	const text = lines.join("\n");
	return {
		text,
		handles,
		chunks,
		tokenCount: Math.round(text.length / 4),
		truncated:
			keptNodes.truncated ||
			keptEdges.truncated ||
			verseRefs.length > MAX_CONTEXT_VERSES,
	};
}

// Checks every citation against the context, then promotes what the answer used: a cited
// handle, and both ends of a cited relationship, become "cited".
// Long answers sometimes repeat their handles inside a sentence ("... [E9, R4]"); they
// count as citations and come out of the text the reader sees.
const INLINE_HANDLES = /\s*\[([ERC]\d+(?:\s*,\s*[ERC]\d+)*)\]/g;

export function applyAnswer(
	raw: RawAnswer,
	handles: Record<string, string>,
	nodes: TraceNode[],
	edges: TraceEdge[],
	chunks: TraceChunk[],
): Answer {
	const invalid = new Set<string>();
	const cited = new Set<string>();
	const sentences = raw.sentences.map(({ text, cites }) => {
		const ids: string[] = [];
		const inline = [...text.matchAll(INLINE_HANDLES)].flatMap((match) =>
			(match[1] ?? "").split(/\s*,\s*/),
		);
		for (const rawHandle of [...cites, ...inline]) {
			const handle = rawHandle.replace(/[[\]\s]/g, "");
			const id = handles[handle];
			if (id === undefined) invalid.add(handle);
			else if (!ids.includes(id)) ids.push(id);
		}
		for (const id of ids) cited.add(id);
		return { text: text.replace(INLINE_HANDLES, "").trim(), cites: ids };
	});
	for (const edge of edges) {
		if (cited.has(edge.id)) {
			cited.add(edge.source);
			cited.add(edge.target);
		}
	}
	for (const element of [...nodes, ...edges, ...chunks]) {
		if (element.status === "context" && cited.has(element.id))
			element.status = "cited";
	}
	return {
		text: sentences.map((sentence) => sentence.text).join(" "),
		sentences,
		invalid_handles: [...invalid],
		uncited_sentence_idx: sentences.flatMap((sentence, index) =>
			sentence.cites.length ? [] : [index],
		),
		insufficient_context: raw.insufficient_context,
	};
}
