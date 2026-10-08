import {
	type AliasHit,
	type EdgeDetail,
	type EntityCard,
	type ExpansionRow,
	type GraphSource,
	type LineageDirection,
	type LinkRow,
	MAX_LINEAGE_DEPTH,
	MAX_LINEAGE_LINKS,
	type Relation,
} from "@kgrag/retrieval";
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

const LINK_RETURN = `RETURN e.id AS edgeId, type(e) AS type, startNode(e).id AS source, endNode(e).id AS target,
			        startNode(e).name AS sourceLabel, endNode(e).name AS targetLabel,
			        e.support_count AS supportCount`;

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

	async lineage(
		personId: string,
		direction: LineageDirection,
		maxDepth: number,
	): Promise<LinkRow[]> {
		// Each link's depth is how many generations its far end lies from the person.
		const path =
			direction === "ancestors"
				? `(:Person)-[:FATHER_OF|MOTHER_OF*1..${MAX_LINEAGE_DEPTH}]->(:Person {id: $personId})`
				: `(:Person {id: $personId})-[:FATHER_OF|MOTHER_OF*1..${MAX_LINEAGE_DEPTH}]->(:Person)`;
		const depthAt = direction === "ancestors" ? "length(p) - i" : "i + 1";
		return this.rows<LinkRow>(
			`MATCH p = ${path}
			 WHERE length(p) <= $maxDepth
			 UNWIND range(0, length(p) - 1) AS i
			 WITH relationships(p)[i] AS e, ${depthAt} AS depth
			 WITH e, min(depth) AS depth
			 ${LINK_RETURN}, depth
			 ORDER BY depth, edgeId LIMIT $limit`,
			{
				personId,
				maxDepth: neo4j.int(Math.min(maxDepth, MAX_LINEAGE_DEPTH)),
				limit: neo4j.int(MAX_LINEAGE_LINKS),
			},
		);
	}

	async relatives(personId: string, relation: Relation): Promise<LinkRow[]> {
		const gender =
			{
				sons: "male",
				brothers: "male",
				daughters: "female",
				sisters: "female",
			}[relation as string] ?? null;
		const direct: Record<Relation, string> = {
			father: "MATCH (:Person)-[e:FATHER_OF]->(:Person {id: $personId})",
			mother: "MATCH (:Person)-[e:MOTHER_OF]->(:Person {id: $personId})",
			parents:
				"MATCH (:Person)-[e:FATHER_OF|MOTHER_OF]->(:Person {id: $personId})",
			children:
				"MATCH (:Person {id: $personId})-[e:FATHER_OF|MOTHER_OF]->(r:Person)",
			sons: "MATCH (:Person {id: $personId})-[e:FATHER_OF|MOTHER_OF]->(r:Person) WHERE r.gender = $gender",
			daughters:
				"MATCH (:Person {id: $personId})-[e:FATHER_OF|MOTHER_OF]->(r:Person) WHERE r.gender = $gender",
			spouses: "MATCH (:Person {id: $personId})-[e:SPOUSE_OF]-(:Person)",
			siblings: "MATCH (:Person {id: $personId})-[e:SIBLING_OF]-(r:Person)",
			brothers:
				"MATCH (:Person {id: $personId})-[e:SIBLING_OF]-(r:Person) WHERE r.gender = $gender",
			sisters:
				"MATCH (:Person {id: $personId})-[e:SIBLING_OF]-(r:Person) WHERE r.gender = $gender",
		};
		const queries = [`${direct[relation]} ${LINK_RETURN}, 1 AS depth`];
		if (
			relation === "siblings" ||
			relation === "brothers" ||
			relation === "sisters"
		) {
			queries.push(
				`MATCH (parent:Person)-[e:FATHER_OF|MOTHER_OF]->(:Person {id: $personId}),
				       (parent)-[:FATHER_OF|MOTHER_OF]->(r:Person)
				 WHERE r.id <> $personId AND ($gender IS NULL OR r.gender = $gender)
				 ${LINK_RETURN}, 1 AS depth`,
				`MATCH (parent:Person)-[:FATHER_OF|MOTHER_OF]->(:Person {id: $personId}),
				       (parent)-[e:FATHER_OF|MOTHER_OF]->(r:Person)
				 WHERE r.id <> $personId AND ($gender IS NULL OR r.gender = $gender)
				 ${LINK_RETURN}, 2 AS depth`,
			);
		}
		const rows = (
			await Promise.all(
				queries.map((query) => this.rows<LinkRow>(query, { personId, gender })),
			)
		).flat();
		const byEdge = new Map<string, LinkRow>();
		for (const row of rows) {
			const known = byEdge.get(row.edgeId);
			if (!known || row.depth < known.depth) byEdge.set(row.edgeId, row);
		}
		return [...byEdge.values()].sort(
			(a, b) => a.depth - b.depth || (a.edgeId < b.edgeId ? -1 : 1),
		);
	}

	async entityCards(ids: string[]): Promise<EntityCard[]> {
		return this.rows<EntityCard>(
			`MATCH (e:Entity) WHERE e.id IN $ids
			 RETURN e.id AS id, e.name AS label, e.kind AS kind, e.gender AS gender,
			        coalesce(e.description, '') AS description, e.first_verse AS firstVerse,
			        e.mention_count AS mentionCount
			 ORDER BY id`,
			{ ids },
		);
	}

	async edgeDetails(ids: string[]): Promise<EdgeDetail[]> {
		return this.rows<EdgeDetail>(
			`MATCH (:Entity)-[e]->(:Entity) WHERE e.id IN $ids
			 RETURN e.id AS id, coalesce(e.description, '') AS description,
			        coalesce(e.evidence_verses, []) AS evidenceVerses, e.other_label AS otherLabel
			 ORDER BY id`,
			{ ids },
		);
	}

	async verses(refs: string[]): Promise<{ id: string; text: string }[]> {
		return this.rows<{ id: string; text: string }>(
			"UNWIND $refs AS ref MATCH (v:Verse {id: ref}) RETURN v.id AS id, v.text AS text",
			{ refs: [...new Set(refs)] },
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
