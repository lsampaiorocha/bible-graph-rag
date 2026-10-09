import { useEffect, useState } from "react";

interface Score {
	precision: number;
	recall: number;
	f1: number;
	gold: number;
}

interface Section {
	entities: Record<string, number | unknown[]>;
	links: Record<string, Score>;
	siblings_with_shared_parents: Score;
	places: Record<string, Score>;
	events: {
		recall: number;
		matched: number;
		gold: number;
		by_size: Record<string, { matched: number; gold: number }>;
		participants: Score;
	};
}

interface Report {
	index: string;
	scope: string;
	corrected_verses: number;
	raw: Section;
	corrected: Section;
	narrative: {
		precision: number;
		interval_95: [number, number];
		labeled: number;
		sampled: number;
	} | null;
}

const ENTITY_ROWS: [string, string, boolean][] = [
	["Entity resolution, B³ F1", "bcubed_f1", true],
	["Merge errors (two people in one node)", "merge_errors", false],
	["Split errors (one person in several nodes)", "split_errors", false],
	["Person mentions F1", "person_mention_f1", true],
	["Place mentions F1", "place_mention_f1", true],
	["Gender accuracy", "gender_accuracy", true],
	["Hard cases passed (of 6)", "hard_cases_passed", false],
];

const pct = (value: number) => `${(value * 100).toFixed(1)}`;

function Table({
	head,
	rows,
}: {
	head: string[];
	rows: (string | number)[][];
}) {
	return (
		<table className="w-full border-collapse text-sm">
			<thead>
				<tr>
					{head.map((cell) => (
						<th
							key={cell}
							className="border-b border-gray-300 px-2 py-1 text-left font-medium text-gray-600"
						>
							{cell}
						</th>
					))}
				</tr>
			</thead>
			<tbody>
				{rows.map((row) => (
					<tr key={String(row[0])} className="border-b border-gray-100">
						{row.map((cell, index) => (
							<td
								key={`${row[0]}-${head[index]}`}
								className={`px-2 py-1 ${index ? "tabular-nums" : ""}`}
							>
								{cell}
							</td>
						))}
					</tr>
				))}
			</tbody>
		</table>
	);
}

interface QaMode {
	accuracy: number;
	verdicts: Record<string, number>;
	by_category: Record<string, number>;
	verse_recall: number | null;
	invalid_citation_rate: number | null;
	mean_input_tokens: number;
	mean_latency_s: number;
	cost_usd: number;
	grounded_rate: number;
	key_people_shown: number | null;
	answer_name_recall: number | null;
	tree_people_recall: number | null;
	tree_people_precision: number | null;
	tree_link_recall: number | null;
	tree_link_precision: number | null;
}

interface QaSummary {
	run: string;
	judge: string;
	questions: number;
	modes: Record<string, QaMode>;
}

const QA_MODES: [string, string][] = [
	["agent", "Graph agent"],
	["vector", "Vector RAG"],
	["closed_book", "LLM from memory"],
];

function QaSection() {
	const [summary, setSummary] = useState<QaSummary | null>(null);

	useEffect(() => {
		fetch("/api/eval/qa")
			.then((response) => (response.ok ? response.json() : null))
			.then(setSummary)
			.catch(() => setSummary(null));
	}, []);

	if (!summary) return null;
	const modes = QA_MODES.filter(([key]) => summary.modes[key]);
	const each = (pick: (mode: QaMode) => string | number) =>
		modes.map(([key]) => pick(summary.modes[key] as QaMode));
	const orDash = (value: number | null) => (value === null ? "–" : pct(value));
	const categories = Object.keys(summary.modes.agent?.by_category ?? {});
	return (
		<section>
			<h3 className="mb-1 font-medium">
				Answer quality on {summary.questions} Genesis questions
			</h3>
			<p className="mb-2 text-sm text-gray-600">
				Same answering model and prompt in every column; only the retrieval
				differs. Graded by {summary.judge} against reference answers (correct =
				1, partial = ½). Family rows cover the lineage questions, graded against
				the answer key&apos;s family tree; the tree rows score the graph the
				page draws, so only the graph agent has them. Verse recall = share of
				the reference verses that reached the model.
			</p>
			<Table
				head={["Measure", ...modes.map(([, label]) => label)]}
				rows={[
					["Accuracy", ...each((mode) => pct(mode.accuracy))],
					...categories.map((category) => [
						`  ${category.replace("_", " ")}`,
						...each((mode) => pct(mode.by_category[category] ?? 0)),
					]),
					[
						"Declined to answer",
						...each((mode) => mode.verdicts.no_answer ?? 0),
					],
					[
						"Sentences backed by their cited sources",
						...each((mode) => pct(mode.grounded_rate)),
					],
					[
						"Key people shown in the graph",
						...each((mode) => orDash(mode.key_people_shown)),
					],
					[
						"Family lists: people named in the answer",
						...each((mode) => orDash(mode.answer_name_recall)),
					],
					[
						"Family tree shown: people found",
						...each((mode) => orDash(mode.tree_people_recall)),
					],
					[
						"Family tree shown: people that belong",
						...each((mode) => orDash(mode.tree_people_precision)),
					],
					[
						"Family tree shown: parent links found",
						...each((mode) => orDash(mode.tree_link_recall)),
					],
					["Verse recall", ...each((mode) => orDash(mode.verse_recall))],
					[
						"Invalid citations",
						...each((mode) => orDash(mode.invalid_citation_rate)),
					],
					["Input tokens (mean)", ...each((mode) => mode.mean_input_tokens)],
					["Latency, s (mean)", ...each((mode) => mode.mean_latency_s)],
					[
						"Cost for all questions",
						...each((mode) => `$${mode.cost_usd.toFixed(3)}`),
					],
				]}
			/>
		</section>
	);
}

export function BenchmarkPage() {
	const [reports, setReports] = useState<Report[] | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		fetch("/api/eval")
			.then(async (response) => {
				const body = await response.json();
				if (!response.ok) throw new Error(body.error);
				setReports(body);
			})
			.catch((reason: Error) => setError(reason.message));
	}, []);

	if (error) return <p className="p-6 text-sm text-red-700">{error}</p>;
	if (!reports)
		return (
			<p className="p-6 text-sm text-gray-500">Loading the latest benchmark…</p>
		);

	const head = ["Measure", ...reports.map((report) => report.index)];
	const each = <T,>(pick: (report: Report) => T) => reports.map(pick);
	return (
		<div className="mx-auto flex max-w-4xl flex-col gap-6 overflow-auto p-6">
			<QaSection />
			<div>
				<h2 className="text-lg font-semibold">
					How the extracted graph scores against an expert-made answer key
				</h2>
				<p className="text-sm text-gray-600">
					Scope {reports[0]?.scope}. The answer key is Theographic Bible
					Metadata, used only for grading. Scores are percentages after{" "}
					{reports[0]?.corrected_verses} documented answer-key corrections; F1
					combines precision and recall.
				</p>
			</div>
			<section>
				<h3 className="mb-1 font-medium">Entities</h3>
				<Table
					head={head}
					rows={ENTITY_ROWS.map(([label, key, isShare]) => [
						label,
						...each((report) => {
							const value = report.corrected.entities[key] as number;
							return isShare ? pct(value) : value;
						}),
					])}
				/>
			</section>
			<section>
				<h3 className="mb-1 font-medium">
					Family links (F1, answer-key count in brackets)
				</h3>
				<Table
					head={head}
					rows={[
						...Object.keys(reports[0]?.corrected.links ?? {}).map((type) => [
							type,
							...each((report) => {
								const score = report.corrected.links[type];
								return score ? `${pct(score.f1)} (${score.gold})` : "";
							}),
						]),
						[
							"SIBLING_OF incl. shared parents",
							...each(
								(report) =>
									`${pct(report.corrected.siblings_with_shared_parents.f1)} (${report.corrected.siblings_with_shared_parents.gold})`,
							),
						],
					]}
				/>
			</section>
			<section>
				<h3 className="mb-1 font-medium">Events</h3>
				<Table
					head={head}
					rows={[
						[
							"Found (recall)",
							...each(
								(r) =>
									`${pct(r.corrected.events.recall)} (${r.corrected.events.matched}/${r.corrected.events.gold})`,
							),
						],
						...Object.keys(reports[0]?.corrected.events.by_size ?? {}).map(
							(label) => [
								`  ${label}`,
								...each((r) => {
									const bucket = r.corrected.events.by_size[label];
									return bucket ? `${bucket.matched}/${bucket.gold}` : "";
								}),
							],
						),
						[
							"Participants F1",
							...each((r) => pct(r.corrected.events.participants.f1)),
						],
					]}
				/>
			</section>
			<section>
				<h3 className="mb-1 font-medium">
					Narrative links (hand-labelled sample)
				</h3>
				<Table
					head={head}
					rows={[
						[
							"Precision (95% interval)",
							...each((r) =>
								r.narrative
									? `${pct(r.narrative.precision)} (${pct(r.narrative.interval_95[0])}–${pct(r.narrative.interval_95[1])}), ${r.narrative.labeled} labelled`
									: "not labelled yet",
							),
						],
					]}
				/>
			</section>
		</div>
	);
}
