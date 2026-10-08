import type { RetrievalTrace } from "@kgrag/retrieval";
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
	const hopById = new Map(trace.nodes.map((node) => [node.id, node.hop]));
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
	const links: GraphLink[] = trace.edges.map((edge) => {
		// Point every link away from the seed so the particles show activation spreading.
		const forward =
			(hopById.get(edge.source) ?? 0) <= (hopById.get(edge.target) ?? 0);
		const [sourceId, targetId] = forward
			? [edge.source, edge.target]
			: [edge.target, edge.source];
		return {
			id: edge.id,
			type: edge.type,
			fromId: edge.source,
			toId: edge.target,
			sourceId,
			targetId,
			source: sourceId,
			target: targetId,
			status: edge.status,
		};
	});
	return {
		nodes,
		links,
		maxHop: Math.max(0, ...nodes.map((node) => node.hop)),
	};
}

function buildWholeGraph(data: GraphData) {
	const nodes: GraphNode[] = data.nodes.map((node) => ({
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
	const seedIds = useMemo(
		() => new Set(trace?.seeds.map((seed) => seed.node_id)),
		[trace],
	);

	useEffect(() => {
		if (!focused) return;
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
	}, [width]);

	const hasAnswer = Boolean(trace?.answer);
	const lineage =
		trace?.steps?.length &&
		trace.steps.every((step) => step.tool === trace.steps?.[0]?.tool)
			? trace.steps[0]?.tool
			: undefined;
	// A family line reads as a tree: ancestors stack upwards, descendants downwards.
	const dagMode =
		mode === "focused"
			? ({ ancestors: "bu", descendants: "td" } as const)[
					lineage as "ancestors" | "descendants"
				]
			: undefined;
	const shown = (status: VizStatus) =>
		status === "cited" || status === "context";
	// Once an answer exists, what it cites stays bright and what was only retrieved dims.
	const dimmed = (status: VizStatus) => hasAnswer && status === "context";
	const isSeed = (node: GraphNode) => seedIds.has(String(node.id));
	const statusOf = (node: GraphNode): VizStatus =>
		node.status ??
		traceNodeStatus.get(String(node.id)) ??
		(trace ? "unreached" : "context");
	const linkStatusOf = (link: GraphLink): VizStatus =>
		link.status ??
		traceEdgeStatus.get(link.id) ??
		(trace ? "unreached" : "context");
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
		<div ref={ref} className="h-full w-full">
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
						}
					}}
					nodeLabel={(node) =>
						`<b>${escapeHtml(node.label)}</b> · ${node.kind}<br/>${describeStatus(node)}`
					}
					nodeCanvasObject={(node, context, scale) => {
						const status = statusOf(node);
						const radius = radiusOf(node);
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
						if (shown(status)) {
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
					nodePointerAreaPaint={(node, color, context) => {
						context.beginPath();
						context.arc(
							node.x ?? 0,
							node.y ?? 0,
							radiusOf(node) + 3,
							0,
							2 * Math.PI,
						);
						context.fillStyle = color;
						context.fill();
					}}
					linkColor={(link) =>
						({
							cited: "#1d4ed8",
							context: hasAnswer ? "#cbd5e1" : "#64748b",
							candidate: "#d6dbe2",
							unreached: "rgba(15,23,42,0.05)",
						})[linkStatusOf(link)]
					}
					linkWidth={(link) =>
						({ cited: 2.2, context: 1.6, candidate: 0.6, unreached: 0.6 })[
							linkStatusOf(link)
						]
					}
					linkLineDash={(link) =>
						linkStatusOf(link) === "candidate" ? [3, 3] : null
					}
					linkLabel={(link) =>
						`${escapeHtml(labelById.get(link.fromId) ?? link.fromId)} —${link.type}→ ${escapeHtml(labelById.get(link.toId) ?? link.toId)}`
					}
					linkDirectionalParticles={(link) =>
						mode === "focused" &&
						linkStatusOf(link) === (hasAnswer ? "cited" : "context")
							? 2
							: 0
					}
					linkDirectionalParticleWidth={2.5}
					linkDirectionalParticleSpeed={0.008}
					onNodeClick={(node) => onSelect(String(node.id))}
					onBackgroundClick={() => onSelect(null)}
				/>
			)}
		</div>
	);
}
