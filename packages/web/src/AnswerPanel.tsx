import type { RetrievalTrace } from "@kgrag/retrieval";

const TOOL_LABELS: Record<string, string> = {
	find_entity: "find entity",
	explore: "explore",
	ancestors: "ancestors",
	descendants: "descendants",
	relatives: "relatives",
	path: "path",
	overview: "overview",
};

interface AnswerPanelProps {
	trace: RetrievalTrace;
	labelById: Map<string, string>;
	onSelect: (id: string) => void;
}

export function AnswerPanel({ trace, labelById, onSelect }: AnswerPanelProps) {
	const edgeById = new Map(trace.edges.map((edge) => [edge.id, edge]));
	const chunkById = new Map(trace.chunks.map((chunk) => [chunk.id, chunk]));
	const label = (id: string) => labelById.get(id) ?? id;

	const citation = (id: string) => {
		const edge = edgeById.get(id);
		if (edge)
			return {
				text: `${label(edge.source)} → ${label(edge.target)}`,
				focus: edge.target,
				kind: "link",
			};
		const chunk = chunkById.get(id);
		if (chunk)
			return {
				text: id,
				focus: chunk.highlight_entity_ids[0] ?? null,
				kind: "verse",
			};
		return { text: label(id), focus: id, kind: "entity" };
	};
	const usage = trace.metrics.usage;
	const funnel = (items: { status: string }[]) => ({
		found: items.length,
		sent: items.filter((item) => item.status !== "candidate").length,
		cited: items.filter((item) => item.status === "cited").length,
	});
	const entities = funnel(trace.nodes);
	const verses = funnel(trace.chunks);
	const seconds =
		Object.values(trace.metrics.timings_ms).reduce((sum, ms) => sum + ms, 0) /
		1000;

	return (
		<div className="flex flex-col gap-3 text-sm">
			<p
				className="text-xs text-gray-600"
				title="found by the search → sent to the LLM → cited in the answer"
			>
				Entities: {entities.found} found → {entities.sent} sent →{" "}
				{entities.cited} cited
				{verses.found > 0 &&
					` · Verses: ${verses.sent} sent → ${verses.cited} cited`}
			</p>
			{trace.steps && trace.steps.length > 0 && (
				<ol className="flex flex-col gap-0.5 rounded bg-gray-50 p-2 font-mono text-xs text-gray-700">
					{trace.steps.map((step) => (
						<li key={step.index}>
							{step.index}. {TOOL_LABELS[step.tool]}(
							{step.entity_id
								? label(step.entity_id)
								: String(step.args.entity)}
							{step.args.relation ? `, ${step.args.relation}` : ""}
							{step.args.other ? `, ${label(String(step.args.other))}` : ""}
							{step.args.family_only ? ", family only" : ""}
							{step.args.generations
								? `, ${step.args.generations} generations`
								: ""}
							) → {step.edge_ids.length} links
							{step.note && (
								<span className="text-amber-700"> · {step.note}</span>
							)}
						</li>
					))}
				</ol>
			)}

			{trace.answer ? (
				<div className="flex flex-col gap-2 leading-relaxed">
					{trace.answer.sentences.map((sentence) => (
						<p
							key={sentence.text}
							className={
								sentence.cites.length
									? ""
									: "underline decoration-dotted decoration-gray-400"
							}
						>
							{sentence.text}{" "}
							{sentence.cites.map((id) => {
								const cite = citation(id);
								return (
									<button
										key={id}
										type="button"
										title={`${cite.kind}: ${cite.text}`}
										onClick={() => cite.focus && onSelect(cite.focus)}
										className="mr-1 rounded bg-blue-50 px-1 text-xs text-blue-800 hover:bg-blue-100"
									>
										{cite.text}
									</button>
								);
							})}
						</p>
					))}
					{trace.answer.insufficient_context && (
						<p className="text-xs text-amber-700">
							The graph did not hold enough to answer fully.
						</p>
					)}
					{trace.answer.invalid_handles.length > 0 && (
						<p className="text-xs text-red-700">
							Dropped {trace.answer.invalid_handles.length} citation(s) to
							things that were not in the context.
						</p>
					)}
				</div>
			) : (
				<p className="text-gray-500">
					{trace.query.params.planner === "keywords"
						? "Retrieval only: keyword planning, no LLM answer. The graph shows what was found."
						: "No answer was generated."}
				</p>
			)}

			{trace.warnings?.map((warning) => (
				<p key={warning} className="text-xs text-amber-700">
					{warning}
				</p>
			))}
			<p className="text-xs text-gray-400">
				{seconds.toFixed(1)} s
				{usage ? ` · ${usage.model} · $${usage.cost_usd.toFixed(4)}` : ""}
			</p>
		</div>
	);
}
