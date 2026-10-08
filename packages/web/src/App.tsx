import type { RetrievalTrace } from "@kgrag/retrieval";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import { AnswerPanel } from "./AnswerPanel.tsx";
import { BenchmarkPage } from "./BenchmarkPage.tsx";
import { EntityPanel } from "./EntityPanel.tsx";
import {
	type GraphData,
	GraphView,
	KIND_COLORS,
	type ViewMode,
} from "./GraphView.tsx";

const EXAMPLE_QUESTIONS = [
	"Who were the ancestors of Noah?",
	"Who were the sons of Noah?",
	"Who did Jacob marry?",
	"What happened at Bethel?",
	"Who was Lot?",
];

type PanelTab = "answer" | "details" | "trace";

export function App() {
	const [graph, setGraph] = useState<GraphData | null>(null);
	const [trace, setTrace] = useState<RetrievalTrace | null>(null);
	const [question, setQuestion] = useState(EXAMPLE_QUESTIONS[0] ?? "");
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const [mode, setMode] = useState<ViewMode>("focused");
	const [selectedId, setSelectedId] = useState<string | null>(null);
	const [tab, setTab] = useState<PanelTab>("answer");
	const [useLlm, setUseLlm] = useState(true);
	const [page, setPage] = useState<"explore" | "benchmark">("explore");
	const [elapsed, setElapsed] = useState(0);
	const labelById = useMemo(
		() => new Map(graph?.nodes.map((node) => [node.id, node.label])),
		[graph],
	);

	useEffect(() => {
		fetch("/api/graph")
			.then((response) =>
				response.ok
					? response.json()
					: Promise.reject(new Error(`graph: HTTP ${response.status}`)),
			)
			.then(setGraph)
			.catch((reason: Error) =>
				setError(
					`${reason.message}. Is the API running? (pnpm --filter @kgrag/api start)`,
				),
			);
	}, []);

	useEffect(() => {
		if (!loading) return;
		setElapsed(0);
		const timer = setInterval(() => setElapsed((seconds) => seconds + 1), 1000);
		return () => clearInterval(timer);
	}, [loading]);

	async function ask(text: string) {
		setQuestion(text);
		setLoading(true);
		setError(null);
		try {
			const response = await fetch("/api/query", {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ text, mode: "agent", llm: useLlm }),
			});
			const body = await response.json();
			if (!response.ok)
				throw new Error(body.error ?? `HTTP ${response.status}`);
			setTrace(body);
			setSelectedId(
				body.steps?.[0]?.entity_id ?? body.seeds[0]?.node_id ?? null,
			);
			setTab("answer");
		} catch (reason) {
			setError(String(reason));
		} finally {
			setLoading(false);
		}
	}

	function submit(event: FormEvent) {
		event.preventDefault();
		void ask(question);
	}

	function select(id: string | null) {
		setSelectedId(id);
		if (id) setTab("details");
	}

	const tabClass = (active: boolean) =>
		`px-2 py-1 text-xs font-medium ${active ? "border-b-2 border-gray-900" : "text-gray-500 hover:text-gray-900"}`;

	return (
		<div className="flex h-full flex-col bg-white text-gray-900">
			<header className="flex items-center gap-4 border-b border-gray-200 px-4 py-2">
				<h1 className="font-semibold">Bible Graph</h1>
				<span className="text-sm text-gray-500">
					{graph
						? `${graph.indexId} · ${graph.nodes.length} entities`
						: "loading…"}
				</span>
				<div className="flex overflow-hidden rounded border border-gray-300 text-xs">
					{(["explore", "benchmark"] as const).map((option) => (
						<button
							key={option}
							type="button"
							onClick={() => setPage(option)}
							className={`px-2 py-1 ${page === option ? "bg-gray-900 text-white" : "hover:bg-gray-100"}`}
						>
							{option === "explore" ? "Explore" : "Benchmark"}
						</button>
					))}
				</div>
				<div className="ml-auto flex items-center gap-4">
					<div className="flex gap-3 text-xs">
						{Object.entries(KIND_COLORS).map(([kind, color]) => (
							<span key={kind} className="flex items-center gap-1">
								<span
									className="inline-block h-2.5 w-2.5 rounded-full"
									style={{ background: color }}
								/>
								{kind}
							</span>
						))}
					</div>
					<div className="flex overflow-hidden rounded border border-gray-300 text-xs">
						{(["focused", "whole"] as const).map((option) => (
							<button
								key={option}
								type="button"
								onClick={() => setMode(option)}
								className={`px-2 py-1 ${mode === option ? "bg-gray-900 text-white" : "hover:bg-gray-100"}`}
							>
								{option === "focused" ? "Focused" : "Whole graph"}
							</button>
						))}
					</div>
				</div>
			</header>
			{page === "benchmark" && <BenchmarkPage />}
			<div
				className={`min-h-0 flex-1 ${page === "explore" ? "flex" : "hidden"}`}
			>
				<aside className="flex w-[440px] shrink-0 flex-col gap-3 border-r border-gray-200 p-4">
					<form onSubmit={submit} className="flex gap-2">
						<input
							value={question}
							onChange={(event) => setQuestion(event.target.value)}
							className="min-w-0 flex-1 rounded border border-gray-300 px-2 py-1.5 text-sm"
							placeholder="Ask about Genesis…"
						/>
						<button
							type="submit"
							disabled={loading || !question.trim()}
							className="rounded bg-gray-900 px-3 py-1.5 text-sm text-white disabled:opacity-50"
						>
							{loading ? "…" : "Ask"}
						</button>
					</form>
					<div className="flex flex-wrap gap-1.5">
						{EXAMPLE_QUESTIONS.map((example) => (
							<button
								key={example}
								type="button"
								disabled={loading}
								onClick={() => void ask(example)}
								className="rounded-full border border-gray-300 px-2 py-0.5 text-xs text-gray-700 hover:bg-gray-100 disabled:opacity-50"
							>
								{example}
							</button>
						))}
					</div>
					<label className="flex items-center gap-2 text-xs text-gray-600">
						<input
							type="checkbox"
							checked={useLlm}
							onChange={(event) => setUseLlm(event.target.checked)}
						/>
						Answer with Claude: plans the graph queries and writes a cited
						answer (about 10 s)
					</label>
					{error && (
						<p className="rounded bg-red-50 p-2 text-sm text-red-700">
							{error}
						</p>
					)}
					{loading && (
						<p className="text-sm text-gray-500">
							{useLlm
								? `Planning, querying the graph, writing a cited answer… ${elapsed} s`
								: "Querying the graph…"}
						</p>
					)}
					<nav className="flex gap-2 border-b border-gray-200">
						{(["answer", "details", "trace"] as const).map((option) => (
							<button
								key={option}
								type="button"
								className={tabClass(tab === option)}
								onClick={() => setTab(option)}
							>
								{option === "answer"
									? "Answer"
									: option === "details"
										? "Details"
										: "Trace JSON"}
							</button>
						))}
					</nav>
					<div className="min-h-0 flex-1 overflow-auto">
						{tab === "answer" &&
							(trace ? (
								<AnswerPanel
									trace={trace}
									labelById={labelById}
									onSelect={select}
								/>
							) : (
								<p className="text-sm text-gray-500">
									Ask a question. The agent picks graph queries, then answers
									with citations you can click.
								</p>
							))}
						{tab === "details" &&
							(selectedId ? (
								<EntityPanel entityId={selectedId} onSelect={select} />
							) : (
								<p className="text-sm text-gray-500">
									Click any node to see its description, relationships and the
									verses where it is named.
								</p>
							))}
						{tab === "trace" && (
							<pre className="rounded bg-gray-50 p-2 text-xs">
								{trace
									? JSON.stringify(trace, null, 2)
									: "Run a query to see its trace."}
							</pre>
						)}
					</div>
				</aside>
				<main className="relative min-w-0 flex-1">
					{graph && (
						<GraphView
							data={graph}
							trace={trace}
							mode={mode}
							selectedId={selectedId}
							onSelect={select}
						/>
					)}
					{mode === "focused" && !trace && (
						<p className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-gray-400">
							Ask a question to watch the graph light up from the entities it
							names.
						</p>
					)}
				</main>
			</div>
		</div>
	);
}
