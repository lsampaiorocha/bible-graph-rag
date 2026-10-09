import { describe, expect, it } from "vitest";
import {
	executeCall,
	MemoryGraphSource,
	planByKeywords,
	type Subgraph,
} from "../src/index.ts";

const entity = (
	id: string,
	name: string,
	kind: string,
	overviewRank?: number,
) => ({
	id,
	name,
	kind,
	mentionCount: 1,
	...(overviewRank ? { overviewRank } : {}),
});
const link = (id: string, type: string, source: string, target: string) => ({
	id,
	type,
	source,
	target,
	supportCount: 1,
	description: "",
	evidenceVerses: [],
});

function source() {
	return new MemoryGraphSource({
		indexId: "test-index",
		entities: [
			entity("per_jacob", "Jacob", "person", 1),
			entity("per_joseph", "Joseph", "person", 2),
			entity("per_dinah", "Dinah", "person"),
			entity("evt_flood", "The flood", "event", 1),
			entity("evt_sold", "Joseph sold into Egypt", "event", 2),
			entity("evt_dream", "Joseph's dream", "event"),
			entity("per_noah", "Noah", "person"),
		],
		edges: [
			link("r1", "FATHER_OF", "per_jacob", "per_joseph"),
			link("r2", "FATHER_OF", "per_jacob", "per_dinah"),
			link("r3", "PARTICIPATED_IN", "per_noah", "evt_flood"),
			link("r4", "PARTICIPATED_IN", "per_joseph", "evt_sold"),
			link("r5", "PARTICIPATED_IN", "per_joseph", "evt_dream"),
		],
		aliases: [],
	});
}

describe("overview tool", () => {
	it("returns the ranked main characters with only the links among them", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source(),
			{ tool: "overview", entity: "people" },
			1,
			subgraph,
		);

		expect(step).toEqual({
			index: 1,
			tool: "overview",
			args: { entity: "people" },
			node_ids: ["per_jacob", "per_joseph"],
			edge_ids: ["r1"],
			ms: expect.any(Number),
		});
		expect([...subgraph.nodes.keys()]).toEqual(["per_jacob", "per_joseph"]);
	});

	it("returns the ranked story events with the main characters who took part", async () => {
		const subgraph: Subgraph = { nodes: new Map(), edges: new Map() };

		const step = await executeCall(
			source(),
			{ tool: "overview", entity: "events" },
			1,
			subgraph,
		);

		expect(step.node_ids).toEqual(["evt_flood", "evt_sold", "per_joseph"]);
		expect(step.edge_ids).toEqual(["r4"]);
	});

	it("is what the keyword planner picks for main-character and main-event questions", () => {
		expect(
			planByKeywords("Who are the main characters of Genesis?", []),
		).toEqual([{ tool: "overview", entity: "people" }]);
		expect(
			planByKeywords("What are the most important events of Genesis?", []),
		).toEqual([{ tool: "overview", entity: "events" }]);
	});
});
