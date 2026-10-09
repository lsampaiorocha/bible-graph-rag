import type { RetrievalTrace } from "@kgrag/retrieval";
import { forceCollide } from "d3-force";
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, {
	type ForceGraphMethods,
	type LinkObject,
	type NodeObject,
} from "react-force-graph-2d";

export interface GraphData {
	indexId: string;
	nodes: {
		id: string;
		label: string;
		kind: string;
		x: number;
		y: number;
		community: number;
		degree: number;
		mentionCount: number;
	}[];
	edges: {
		id: string;
		source: string;
		target: string;
		type: string;
		supportCount: number;
	}[];
}

export type ViewMode = "focused" | "whole";
type VizStatus = "cited" | "context" | "candidate" | "unreached";

interface VizNode {
	id: string;
	label: string;
	kind: string;
	hop: number;
	seed: boolean;
	parentId?: string;
	status?: VizStatus;
	hub?: boolean;
}

interface VizLink {
	id: string;
	type: string;
	fromId: string;
	toId: string;
	sourceId: string;
	targetId: string;
	status?: VizStatus;
}

type GraphNode = NodeObject<VizNode>;
type GraphLink = LinkObject<VizNode, VizLink>;

export const KIND_COLORS: Record<string, string> = {
	person: "#2563eb",
	place: "#16a34a",
	group: "#9333ea",
	event: "#d97706",
};
const CANDIDATE_COLOR = "#cbd5e1";
const UNREACHED_COLOR = "#eef0f3";
const REVEAL_STEP_MS = 700;
const MAX_REVEAL_MS = 3000;
const WHOLE_GRAPH_SCALE = 4;
const MAX_LABEL_LENGTH = 24;

function shortLabel(label: string): string {
	return label.length > MAX_LABEL_LENGTH
		? `${label.slice(0, MAX_LABEL_LENGTH - 1)}…`
		: label;
}

function escapeHtml(text: string): string {
	return text.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

function useElementSize<T extends HTMLElement>() {
	const ref = useRef<T>(null);
	const [size, setSize] = useState({ width: 0, height: 0 });
	useEffect(() => {
		if (!ref.current) return;
		const observer = new ResizeObserver(([entry]) => {
			if (entry)
				setSize({
					width: entry.contentRect.width,
					height: entry.contentRect.height,
				});
		});
		observer.observe(ref.current);
		return () => observer.disconnect();
	}, []);
	return { ref, ...size };
}

function buildFocusedGraph(
	trace: RetrievalTrace,
	kindById: Map<string, string>,
) {
	const edgeById = new Map(trace.edges.map((edge) => [edge.id, edge]));
	const seedIds = new Set(trace.seeds.map((seed) => seed.node_id));
	const nodes: GraphNode[] = trace.nodes.map((node) => {
		const via = node.reached_via ? edgeById.get(node.reached_via) : undefined;
		return {
			id: node.id,
			label: node.label ?? node.id,
			kind: kindById.get(node.id) ?? "event",
			hop: node.hop,
			seed: seedIds.has(node.id),
			parentId: via
				? via.source === node.id
					? via.target
					: via.source
				: undefined,
			status: node.status,
		};
	});
	const links: GraphLink[] = trace.edges.map((edge) => ({
		id: edge.id,
		type: edge.type,
		fromId: edge.source,
		toId: edge.target,
		sourceId: edge.source,
		targetId: edge.target,
		source: edge.source,
		target: edge.target,
		status: edge.status,
	}));
	return {
		nodes,
		links,
		maxHop: Math.max(0, ...nodes.map((node) => node.hop)),
	};
}

const WHOLE_GRAPH_LABELS = 30;
// Two-way family ties get a line style instead of an arrow; every other link points
// from its source to its target (parent to child, person to event, event to place).
const LINK_DASH: Record<string, number[]> = {
	SPOUSE_OF: [6, 4],
	SIBLING_OF: [1.5, 3],
};
const RADIAL_TREE_NODES = 40;
// The tightest zoom the label spacing adapts to; past it a wide tree would only keep
// widening, as more spacing makes the fit zoom out further.
const MIN_LABEL_SCALE = 0.5;

function buildWholeGraph(data: GraphData) {
	const hubs = new Set(
		[...data.nodes]
			.sort((a, b) => b.mentionCount - a.mentionCount)
			.slice(0, WHOLE_GRAPH_LABELS)
			.map((node) => node.id),
	);
	const nodes: GraphNode[] = data.nodes.map((node) => ({
		hub: hubs.has(node.id),
		id: node.id,
		label: node.label,
		kind: node.kind,
		hop: 0,
		seed: false,
		fx: node.x * WHOLE_GRAPH_SCALE,
		fy: node.y * WHOLE_GRAPH_SCALE,
	}));
	const links: GraphLink[] = data.edges.map((edge) => ({
		id: edge.id,
		type: edge.type,
		fromId: edge.source,
		toId: edge.target,
		sourceId: edge.source,
		targetId: edge.target,
		source: edge.source,
		target: edge.target,
	}));
	return { nodes, links };
}

interface GraphViewProps {
	data: GraphData;
	trace: RetrievalTrace | null;
	mode: ViewMode;
	selectedId: string | null;
	onSelect: (id: string | null) => void;
}

export function GraphView({
	data,
	trace,
	mode,
	selectedId,
	onSelect,
}: GraphViewProps) {
	const { ref, width, height } = useElementSize<HTMLDivElement>();
	const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(
		undefined,
	);
	const needsFit = useRef(true);
	// Labels keep a fixed size on screen, so how far apart nodes must sit (in graph units)
	// depends on the zoom the fit lands on; each fit rescales the collision once or twice.
	const labelScale = useRef(1);
	const labelPasses = useRef(0);
	const setCollide = (scale: number) =>
		graphRef.current?.d3Force(
			"collide",
			forceCollide<GraphNode>(
				(node) =>
					(Math.max(8, shortLabel(node.label).length * 3.6) + 2) / scale,
			),
		);
	const [revealedHop, setRevealedHop] = useState(0);

	const kindById = useMemo(
		() => new Map(data.nodes.map((node) => [node.id, node.kind])),
		[data],
	);
	const labelById = useMemo(
		() => new Map(data.nodes.map((node) => [node.id, node.label])),
		[data],
	);
	const whole = useMemo(() => buildWholeGraph(data), [data]);
	const focused = useMemo(
		() => (trace ? buildFocusedGraph(trace, kindById) : null),
		[trace, kindById],
	);
	const traceNodeStatus = useMemo(
		() =>
			new Map<string, VizStatus>(
				trace?.nodes.map((node) => [node.id, node.status]),
			),
		[trace],
	);
	const traceEdgeStatus = useMemo(
		() =>
			new Map<string, VizStatus>(
				trace?.edges.map((edge) => [edge.id, edge.status]),
			),
		[trace],
	);
	// What the answer uses: what it cites, plus anyone it names. The model often cites a
	// verse rather than a person ("Leah bore Reuben" citing Gen 35:23), and that person
	// should light up too.
	const usedIds = useMemo(() => {
		const used = new Set(
			trace?.nodes
				.filter((node) => node.status === "cited")
				.map((node) => node.id),
		);
		const text =
			trace?.answer?.sentences.map((sentence) => sentence.text).join(" ") ?? "";
		for (const node of trace?.nodes ?? []) {
			const label = node.label?.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
			if (label && new RegExp(`(?<![\\w'])${label}(?!\\w)`).test(text))
				used.add(node.id);
		}
		return used;
	}, [trace]);
	// Only seeds a step used are ringed: an overview question matches names it never uses.
	const seedIds = useMemo(() => {
		const used = new Set(trace?.steps?.map((step) => step.entity_id));
		return new Set(
			trace?.seeds
				.map((seed) => seed.node_id)
				.filter((id) => !trace.steps?.length || used.has(id)),
		);
	}, [trace]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: setCollide only reads refs.
	useEffect(() => {
		if (!focused) return;
		labelScale.current = 1;
		labelPasses.current = 0;
		setCollide(1);
		setRevealedHop(0);
		const timers = Array.from({ length: focused.maxHop }, (_, index) =>
			setTimeout(
				() => setRevealedHop(index + 1),
				Math.min(REVEAL_STEP_MS, MAX_REVEAL_MS / focused.maxHop) * (index + 1),
			),
		);
		return () => timers.forEach(clearTimeout);
	}, [focused]);

	const graphData = useMemo(() => {
		if (mode === "whole") return whole;
		if (!focused) return { nodes: [], links: [] };
		const byId = new Map(focused.nodes.map((node) => [node.id, node]));
		const visible = focused.nodes.filter((node) => node.hop <= revealedHop);
		for (const node of visible) {
			const parent = node.parentId ? byId.get(node.parentId) : undefined;
			if (
				node.x === undefined &&
				parent?.x !== undefined &&
				parent.y !== undefined
			) {
				// New nodes start next to the node that reached them, so they fly out from it.
				const angle = (node.id.length * 2.399) % (2 * Math.PI);
				node.x = parent.x + 12 * Math.cos(angle);
				node.y = parent.y + 12 * Math.sin(angle);
			}
		}
		const visibleIds = new Set(visible.map((node) => node.id));
		return {
			nodes: visible,
			links: focused.links.filter(
				(link) =>
					visibleIds.has(link.sourceId) && visibleIds.has(link.targetId),
			),
		};
	}, [mode, whole, focused, revealedHop]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: graphData is the trigger, so the view refits whenever the displayed nodes change.
	useEffect(() => {
		needsFit.current = true;
		if (mode !== "whole") {
			// Refit soon after each reveal too: waiting for the physics to settle leaves
			// the newest generations off-screen for several seconds.
			const timer = setTimeout(() => graphRef.current?.zoomToFit(400, 60), 450);
			return () => clearTimeout(timer);
		}
		const focusIds = new Set(
			trace?.nodes
				.filter((node) => node.status !== "candidate")
				.map((node) => node.id),
		);
		const timer = setTimeout(
			() =>
				graphRef.current?.zoomToFit(
					600,
					40,
					(node) => focusIds.size === 0 || focusIds.has(String(node.id)),
				),
			50,
		);
		return () => clearTimeout(timer);
	}, [graphData, mode, trace]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: width is the trigger, because the graph only mounts once its container has a size.
	useEffect(() => {
		// Stronger repulsion and longer links than the defaults keep labels from overlapping.
		graphRef.current?.d3Force("charge")?.strength(-160);
		graphRef.current?.d3Force("link")?.distance(45);
		setCollide(labelScale.current);
	}, [width]);

	const hasAnswer = Boolean(trace?.answer);
	const lineage =
		trace?.steps?.length &&
		trace.steps.every((step) => step.tool === trace.steps?.[0]?.tool)
			? trace.steps[0]?.tool
			: undefined;
	// A family line reads as a tree: ancestors stack upwards, descendants downwards.
	// A family line reads as a tree with parents above children; a wide descendant tree
	// fits far better as rings around the ancestor.
	const dagMode =
		mode !== "focused" || (lineage !== "ancestors" && lineage !== "descendants")
			? undefined
			: lineage === "descendants" &&
					(focused?.nodes.length ?? 0) > RADIAL_TREE_NODES
				? "radialout"
				: "td";
	const shown = (status: VizStatus) =>
		status === "cited" || status === "context";
	// Once an answer exists, what it uses stays bright and what was only retrieved dims.
	const dimmed = (status: VizStatus) => hasAnswer && status === "context";
	const isSeed = (node: GraphNode) => seedIds.has(String(node.id));
	const statusOf = (node: GraphNode): VizStatus => {
		const status =
			node.status ??
			traceNodeStatus.get(String(node.id)) ??
			(trace ? "unreached" : "context");
		return status === "context" && usedIds.has(String(node.id))
			? "cited"
			: status;
	};
	const linkStatusOf = (link: GraphLink): VizStatus => {
		const status =
			link.status ??
			traceEdgeStatus.get(link.id) ??
			(trace ? "unreached" : "context");
		return status === "context" &&
			usedIds.has(link.fromId) &&
			usedIds.has(link.toId)
			? "cited"
			: status;
	};
	const linkColorOf = (link: GraphLink) =>
		({
			cited: "#1d4ed8",
			context: hasAnswer ? "#cbd5e1" : "#64748b",
			candidate: "#d6dbe2",
			unreached: "rgba(15,23,42,0.05)",
		})[linkStatusOf(link)];
	const tiers = { used: 0, sent: 0, found: 0 };
	for (const node of focused?.nodes ?? []) {
		const status = statusOf(node);
		if (status === "cited") tiers.used += 1;
		else if (status === "context") tiers.sent += 1;
		else if (status === "candidate") tiers.found += 1;
	}
	const radiusOf = (node: GraphNode) => {
		if (isSeed(node)) return 9;
		const status = statusOf(node);
		if (status === "candidate") return 3.5;
		if (status === "unreached") return 2;
		return mode === "whole" ? 4 : 6 - Math.min(node.hop, 2);
	};
	const describeStatus = (node: GraphNode) => {
		const status = statusOf(node);
		if (status === "candidate") return "dropped by the neighbour cap";
		if (status === "unreached") return "not reached by this query";
		if (status === "cited") return `cited in the answer · hop ${node.hop}`;
		return trace
			? `${hasAnswer ? "retrieved, not cited" : "kept"} · hop ${node.hop}`
			: "";
	};

	return (
		<div className="flex h-full w-full flex-col">
			<div ref={ref} className="min-h-0 flex-1">
				{width > 0 && (
					<ForceGraph2D<VizNode, VizLink>
						ref={graphRef}
						width={width}
						height={height}
						graphData={graphData}
						cooldownTicks={mode === "whole" ? 0 : 200}
						dagMode={dagMode}
						dagLevelDistance={55}
						onDagError={() => undefined}
						d3VelocityDecay={0.35}
						onEngineStop={() => {
							if (mode === "focused" && needsFit.current) {
								needsFit.current = false;
								graphRef.current?.zoomToFit(500, 60);
								setTimeout(() => {
									const target = Math.max(
										graphRef.current?.zoom() ?? 1,
										MIN_LABEL_SCALE,
									);
									if (
										labelPasses.current < 2 &&
										Math.abs(target - labelScale.current) / labelScale.current >
											0.12
									) {
										labelPasses.current += 1;
										labelScale.current = target;
										setCollide(labelScale.current);
										needsFit.current = true;
										graphRef.current?.d3ReheatSimulation();
									}
								}, 550);
							}
						}}
						nodeLabel={(node) =>
							`<b>${escapeHtml(node.label)}</b> · ${node.kind}<br/>${describeStatus(node)}`
						}
						nodeCanvasObject={(node, context, scale) => {
							const status = statusOf(node);
							// Circles stop growing past 1.5x zoom so small graphs do not show giant dots.
							const radius = radiusOf(node) * Math.min(1, 1.5 / scale);
							const x = node.x ?? 0;
							const y = node.y ?? 0;
							context.globalAlpha = dimmed(status) ? 0.35 : 1;
							context.beginPath();
							context.arc(x, y, radius, 0, 2 * Math.PI);
							context.fillStyle =
								status === "candidate"
									? CANDIDATE_COLOR
									: status === "unreached"
										? UNREACHED_COLOR
										: (KIND_COLORS[node.kind] ?? "#6b7280");
							context.fill();
							if (isSeed(node) || node.id === selectedId) {
								context.lineWidth = 2.5 / scale;
								context.strokeStyle =
									node.id === selectedId ? "#f43f5e" : "#111827";
								context.stroke();
							}
							// The whole graph names only its hubs until you zoom in.
							if (
								shown(status) &&
								(mode !== "whole" || node.hub || scale > 3)
							) {
								const label = shortLabel(node.label);
								const fontSize = 12 / scale;
								context.font = `${isSeed(node) || status === "cited" ? "600 " : ""}${fontSize}px ui-sans-serif, system-ui, sans-serif`;
								context.textAlign = "center";
								context.textBaseline = "top";
								context.lineWidth = 3 / scale;
								context.strokeStyle = "white";
								context.strokeText(label, x, y + radius + 2 / scale);
								context.fillStyle = "#111827";
								context.fillText(label, x, y + radius + 2 / scale);
							}
							context.globalAlpha = 1;
						}}
						nodePointerAreaPaint={(node, color, context, scale) => {
							context.beginPath();
							context.arc(
								node.x ?? 0,
								node.y ?? 0,
								radiusOf(node) * Math.min(1, 1.5 / scale) + 3,
								0,
								2 * Math.PI,
							);
							context.fillStyle = color;
							context.fill();
						}}
						linkColor={linkColorOf}
						linkWidth={(link) =>
							({ cited: 2.2, context: 1.6, candidate: 0.6, unreached: 0.6 })[
								linkStatusOf(link)
							]
						}
						linkLineDash={(link) => LINK_DASH[link.type] ?? null}
						linkDirectionalArrowLength={(link) =>
							LINK_DASH[link.type] ? 0 : 5
						}
						linkDirectionalArrowRelPos={1}
						linkDirectionalArrowColor={linkColorOf}
						nodeRelSize={6}
						linkLabel={(link) =>
							`${escapeHtml(labelById.get(link.fromId) ?? link.fromId)} —${link.type}→ ${escapeHtml(labelById.get(link.toId) ?? link.toId)}`
						}
						onNodeClick={(node) => onSelect(String(node.id))}
						onBackgroundClick={() => onSelect(null)}
					/>
				)}
			</div>
			{mode === "focused" && trace && (
				<div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-gray-200 bg-white px-3 py-2 text-[11px] text-gray-700">
					<div className="font-semibold text-gray-900">
						How to read this graph
					</div>
					{hasAnswer && (
						<div className="flex items-center gap-2">
							<svg width="14" height="14" aria-hidden="true">
								<circle cx="7" cy="7" r="5" fill="#2563eb" />
							</svg>
							<span>
								<b>Used in the answer</b>, cited or named · {tiers.used}
							</span>
						</div>
					)}
					<div className="flex items-center gap-2">
						<svg width="14" height="14" aria-hidden="true">
							<circle
								cx="7"
								cy="7"
								r="5"
								fill="#2563eb"
								opacity={hasAnswer ? 0.35 : 1}
							/>
						</svg>
						<span>
							{hasAnswer ? "Sent to the model, not used" : "Sent to the model"}{" "}
							· {tiers.sent}
						</span>
					</div>
					<div className="flex items-center gap-2">
						<svg width="14" height="14" aria-hidden="true">
							<circle cx="7" cy="7" r="3" fill={CANDIDATE_COLOR} />
						</svg>
						<span>
							Found, not sent (cut by ranking or limits) · {tiers.found}
						</span>
					</div>
					<div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-gray-600">
						<span className="flex items-center gap-1">
							<svg width="26" height="10" aria-hidden="true">
								<line
									x1="1"
									y1="5"
									x2="19"
									y2="5"
									stroke="#1d4ed8"
									strokeWidth="1.6"
								/>
								<path d="M19 1.5 L25 5 L19 8.5 Z" fill="#1d4ed8" />
							</svg>
							parent to child, who or what to which
						</span>
						<span className="flex items-center gap-1">
							<svg width="22" height="10" aria-hidden="true">
								<line
									x1="1"
									y1="5"
									x2="21"
									y2="5"
									stroke="#1d4ed8"
									strokeWidth="1.6"
									strokeDasharray="6 4"
								/>
							</svg>
							married
						</span>
						<span className="flex items-center gap-1">
							<svg width="22" height="10" aria-hidden="true">
								<line
									x1="1"
									y1="5"
									x2="21"
									y2="5"
									stroke="#1d4ed8"
									strokeWidth="1.6"
									strokeDasharray="1.5 3"
									strokeLinecap="round"
								/>
							</svg>
							siblings
						</span>
						<span>colour = kind, see the top bar</span>
					</div>
				</div>
			)}
		</div>
	);
}
