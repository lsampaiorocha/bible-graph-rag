import os
import re
from collections.abc import Iterator

from neo4j import Driver, GraphDatabase, Session

BATCH_SIZE = 1000
EMBEDDING_DIMENSIONS = 384
NODE_LABELS = {"person": "Person", "place": "Place", "group": "Group", "event": "Event"}

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT alias_key IF NOT EXISTS FOR (a:Alias) REQUIRE a.key IS UNIQUE",
    *(
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.embedding "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {EMBEDDING_DIMENSIONS}, "
        "`vector.similarity_function`: 'cosine' } }"
        for name, label in (
            ("entity_embedding", "Entity"),
            ("passage_embedding", "Passage"),
        )
    ),
]


def alias_key(name: str) -> str:
    # The TypeScript seed lookup builds the same key from question n-grams.
    name = re.sub(r"^the\s+", "", name.strip().lower()).replace("’", "'")
    name = re.sub(r"'s\b", "", name)
    name = re.sub(r"['\-]", "", name)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name).split())


def connect() -> Driver:
    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    return GraphDatabase.driver(uri, auth=("neo4j", os.environ["NEO4J_PASSWORD"]))


def batches(rows: list, size: int = BATCH_SIZE) -> Iterator[list]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def write_rows(session: Session, query: str, rows: list[dict]) -> None:
    for batch in batches(rows):
        session.run(query, rows=batch).consume()


def without_nulls(properties: dict) -> dict:
    return {key: value for key, value in properties.items() if value is not None}


def load_graph(
    driver: Driver,
    meta: dict,
    verses: list[dict],
    passages: list[dict],
    entities: list[dict],
    edges: list[dict],
    entity_vectors: dict[str, list[float]],
    passage_vectors: dict[str, list[float]],
) -> None:
    verse_rows = [
        {
            "id": verse["osis_ref"],
            "verse_id": verse["verse_id"],
            "book": verse["book"],
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "ordinal": ordinal,
            "text": verse["text"],
        }
        for ordinal, verse in enumerate(verses)
    ]
    passage_rows = [
        {
            "id": passage["id"],
            "book": passage["book"],
            "chapter": passage["chapter"],
            "first_verse": passage["verse_refs"][0],
            "last_verse": passage["verse_refs"][-1],
            "text": passage["text"],
            "token_count": passage["token_count"],
            "embedding": passage_vectors[passage["id"]],
        }
        for passage in passages
    ]
    entity_rows = [
        {
            "label": NODE_LABELS[entity["kind"]],
            "props": without_nulls(
                {
                    "id": entity["id"],
                    "kind": entity["kind"],
                    "name": entity["name"],
                    "aliases": entity["aliases"],
                    "description": entity["description"],
                    "mention_count": entity["mention_count"],
                    "first_verse": entity["first_verse"],
                    "degree": entity["degree"],
                    "community": entity["community"],
                    "x": entity["x"],
                    "y": entity["y"],
                    "gender": entity.get("gender"),
                    "place_kind": entity.get("place_kind"),
                    "overview_rank": entity.get("overview_rank"),
                    "embedding": entity_vectors[entity["id"]],
                }
            ),
        }
        for entity in entities
    ]
    edge_rows = [
        {
            "type": edge["type"],
            "source": edge["source"],
            "target": edge["target"],
            "props": without_nulls(
                {
                    "id": edge["id"],
                    "description": edge["description"],
                    "evidence_verses": edge["evidence_verses"],
                    "evidence_passages": edge["evidence_passages"],
                    "support_count": edge["support_count"],
                    "other_label": edge["other_label"],
                    "inferred": edge.get("inferred"),
                }
            ),
        }
        for edge in edges
    ]
    alias_rows = sorted(
        {
            (alias_key(name), entity["id"])
            for entity in entities
            if entity["kind"] != "event"
            for name in [entity["name"], *entity["aliases"]]
            if alias_key(name)
        }
    )

    with driver.session() as session:
        session.run(
            "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()
        for statement in SCHEMA_STATEMENTS:
            session.run(statement).consume()
        write_rows(
            session, "UNWIND $rows AS row CREATE (v:Verse) SET v = row", verse_rows
        )
        write_rows(
            session, "UNWIND $rows AS row CREATE (p:Passage) SET p = row", passage_rows
        )
        write_rows(
            session,
            "UNWIND $rows AS row MATCH (p:Passage {id: row.passage}) MATCH (v:Verse {id: row.verse}) "
            "CREATE (p)-[:CONTAINS]->(v)",
            [
                {"passage": passage["id"], "verse": verse}
                for passage in passages
                for verse in passage["verse_refs"]
            ],
        )
        write_rows(
            session,
            "UNWIND $rows AS row CREATE (e:Entity:$(row.label)) SET e = row.props",
            entity_rows,
        )
        write_rows(
            session,
            "UNWIND $rows AS row MATCH (e:Entity {id: row.entity}) MATCH (v:Verse {id: row.verse}) "
            "CREATE (e)-[:MENTIONED_IN]->(v)",
            [
                {"entity": entity["id"], "verse": verse}
                for entity in entities
                for verse in entity["mention_verses"]
            ],
        )
        write_rows(
            session,
            "UNWIND $rows AS row MATCH (a:Entity {id: row.source}) MATCH (b:Entity {id: row.target}) "
            "CREATE (a)-[r:$(row.type)]->(b) SET r = row.props",
            edge_rows,
        )
        write_rows(
            session,
            "UNWIND $rows AS row MERGE (a:Alias {key: row.key}) WITH a, row "
            "MATCH (e:Entity {id: row.entity}) CREATE (a)-[:ALIAS_OF]->(e)",
            [{"key": key, "entity": entity_id} for key, entity_id in alias_rows],
        )
        session.run("CREATE (m:Meta) SET m = $meta", meta=meta).consume()
        session.run("CALL db.awaitIndexes(300)").consume()


def graph_counts(driver: Driver) -> dict[str, int]:
    with driver.session() as session:
        node_counts = {
            record["label"]: record["count"]
            for record in session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label"
            )
        }
        relationship_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        ).single()["count"]
        online_vector_indexes = session.run(
            "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR' AND state = 'ONLINE' RETURN count(*) AS count"
        ).single()["count"]
    return {
        **node_counts,
        "relationships": relationship_count,
        "online_vector_indexes": online_vector_indexes,
    }
