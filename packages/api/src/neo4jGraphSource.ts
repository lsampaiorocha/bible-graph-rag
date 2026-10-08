import type { AliasHit, ExpansionRow, GraphSource } from "@kgrag/retrieval";
import neo4j, { type Driver } from "neo4j-driver";

export interface GraphView {
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

export interface EntityDetails {
	id: string;
	name: string;
	kind: string;
	aliases: string[];
	description: string;
	gender: string | null;
	placeKind: string | null;
	mentionCount: number;
	verses: { id: string; text: string }[];
	relations: {
		id: string;
		type: string;
		outgoing: boolean;
		neighbourId: string;
		neighbourName: string;
		description: string;
		otherLabel: string | null;
		verses: string[];
		supportCount: number;
	}[];
}

export class Neo4jGraphSource implements GraphSource {
	private readonly driver: Driver;

	constructor(driver: Driver) {
		this.driver = driver;
	}

	private async rows<T>(
		query: string,
		parameters: Record<string, unknown> = {},
	): Promise<T[]> {
		const result = await this.driver.executeQuery(query, parameters, {
			routing: neo4j.routing.READ,
		});
		return result.records.map((record) => record.toObject() as T);
	}

	async indexId(): Promise<string> {
		const [meta] = await this.rows<{ indexId: string }>(
			"MATCH (m:Meta) RETURN m.index_id AS indexId",
		);
		if (!meta)
			throw new Error(
				"Neo4j holds no index; run `uv run kgrag neo4j-load` first",
			);
		return meta.indexId;
	}

	async seedsByAlias(keys: string[]): Promise<AliasHit[]> {
		return this.rows<AliasHit>(
			`UNWIND $keys AS key
			 MATCH (:Alias {key: key})-[:ALIAS_OF]->(e:Entity)
			 RETURN key, e.id AS nodeId, e.name AS label, e.mention_count AS mentionCount
			 ORDER BY key, mentionCount DESC, nodeId`,
			{ keys },
		);
	}

	async expand(
		frontierIds: string[],
		fetchPerNode: number,
	): Promise<ExpansionRow[]> {
		return this.rows<ExpansionRow>(
			`UNWIND $frontierIds AS fromId
			 MATCH (f:Entity {id: fromId})
			 CALL (f) {
			   MATCH (f)-[r]-(n:Entity)
			   RETURN r, n ORDER BY r.support_count DESC, r.id ASC LIMIT $fetchPerNode
			 }
			 RETURN fromId, r.id AS edgeId, type(r) AS type, startNode(r).id AS source,
			        endNode(r).id AS target, r.support_count AS supportCount,
			        n.id AS neighbourId, n.name AS neighbourLabel
			 ORDER BY fromId, supportCount DESC, edgeId`,
			{ frontierIds, fetchPerNode: neo4j.int(fetchPerNode) },
		);
	}

	async graphView(): Promise<GraphView> {
		const [nodes, edges] = await Promise.all([
			this.rows<GraphView["nodes"][number]>(
				`MATCH (e:Entity)
				 RETURN e.id AS id, e.name AS label, e.kind AS kind, e.x AS x, e.y AS y,
				        e.community AS community, e.degree AS degree, e.mention_count AS mentionCount
				 ORDER BY id`,
			),
			this.rows<GraphView["edges"][number]>(
				`MATCH (a:Entity)-[r]->(b:Entity)
				 RETURN r.id AS id, a.id AS source, b.id AS target, type(r) AS type,
				        r.support_count AS supportCount
				 ORDER BY id`,
			),
		]);
		return { indexId: await this.indexId(), nodes, edges };
	}

	async entityDetails(id: string): Promise<EntityDetails | null> {
		const [details] = await this.rows<EntityDetails>(
			`MATCH (e:Entity {id: $id})
			 CALL (e) {
			   MATCH (e)-[:MENTIONED_IN]->(v:Verse)
			   WITH v ORDER BY v.ordinal LIMIT 20
			   RETURN collect(v {.id, .text}) AS verses
			 }
			 CALL (e) {
			   MATCH (e)-[r]-(n:Entity)
			   WITH r, n ORDER BY r.support_count DESC, r.id LIMIT 30
			   RETURN collect({
			     id: r.id, type: type(r), outgoing: startNode(r) = e, neighbourId: n.id,
			     neighbourName: n.name, description: r.description, otherLabel: r.other_label,
			     verses: r.evidence_verses, supportCount: r.support_count
			   }) AS relations
			 }
			 RETURN e.id AS id, e.name AS name, e.kind AS kind, e.aliases AS aliases,
			        e.description AS description, e.gender AS gender, e.place_kind AS placeKind,
			        e.mention_count AS mentionCount, verses, relations`,
			{ id },
		);
		return details ?? null;
	}
}
