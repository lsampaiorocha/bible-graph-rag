import hashlib
from collections import Counter

import networkx as nx

from kgrag.corpus import DATA_DIR
from kgrag.resolve import OccurrenceKey
from kgrag.schema import GROUP, PERSON, PLACE, RELATION_DOMAINS

SYMMETRIC_TYPES = {"SIBLING_OF", "SPOUSE_OF"}
GRAPH_SEED = 42


def edge_id(edge_type: str, source: str, target: str, other_label: str | None) -> str:
    key = f"{edge_type}|{source}|{target}|{other_label or ''}"
    return f"rel_{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def resolve_endpoint(
    resolution: dict[OccurrenceKey, str],
    passage_id: str,
    name: str,
    allowed_kinds: frozenset[str],
) -> str | None:
    for kind in (PERSON, PLACE, GROUP):
        if kind in allowed_kinds and (passage_id, kind, name) in resolution:
            return resolution[(passage_id, kind, name)]
    return None


def build_index(
    extractions: list[dict],
    resolution: dict[OccurrenceKey, str],
    verse_order: dict[str, int],
    layout: bool = True,
) -> tuple[list[dict], list[dict]]:
    entities: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}

    def add_entity(
        passage_id: str, kind: str, name: str, record: dict, verses: list[str]
    ) -> str:
        entity_key = resolution[(passage_id, kind, name)]
        entity = entities.setdefault(
            entity_key,
            {
                "id": entity_key,
                "kind": kind,
                "names": Counter(),
                "aliases": set(),
                "descriptions": [],
                "genders": Counter(),
                "place_kinds": Counter(),
                "mention_verses": set(),
                "passages": set(),
            },
        )
        entity["names"][name] += 1
        entity["aliases"].update(record.get("aliases", []))
        if record.get("description"):
            entity["descriptions"].append(record["description"])
        if "gender" in record:
            entity["genders"][record["gender"]] += 1
        if "kind" in record:
            entity["place_kinds"][record["kind"]] += 1
        entity["mention_verses"].update(verses)
        entity["passages"].add(passage_id)
        return entity_key

    def add_edge(
        edge_type: str,
        source: str,
        target: str,
        description: str,
        verses: list[str],
        passage_id: str,
        other_label: str | None = None,
    ) -> None:
        if edge_type in SYMMETRIC_TYPES:
            source, target = sorted((source, target))
        edge = edges.setdefault(
            (edge_type, source, target, other_label),
            {
                "id": edge_id(edge_type, source, target, other_label),
                "type": edge_type,
                "source": source,
                "target": target,
                "other_label": other_label,
                "descriptions": [],
                "evidence_verses": set(),
                "evidence_passages": set(),
            },
        )
        edge["descriptions"].append(description)
        edge["evidence_verses"].update(verses)
        edge["evidence_passages"].add(passage_id)

    for extraction in extractions:
        passage_id = extraction["passage_id"]
        validated = extraction["validated"]
        for kind, records in (
            (PERSON, validated["people"]),
            (PLACE, validated["places"]),
            (GROUP, validated["groups"]),
        ):
            for record in records:
                add_entity(passage_id, kind, record["name"], record, record["named_in"])
        for event in validated["events"]:
            event_key = add_entity(
                passage_id, "event", event["title"], {}, event["verses"]
            )
            for participant in event["participants"]:
                participant_key = resolve_endpoint(
                    resolution, passage_id, participant, frozenset({PERSON, GROUP})
                )
                if participant_key is not None:
                    add_edge(
                        "PARTICIPATED_IN",
                        participant_key,
                        event_key,
                        f"{participant} took part in {event['title']}.",
                        event["verses"],
                        passage_id,
                    )
            if event["place"]:
                place_key = resolve_endpoint(
                    resolution, passage_id, event["place"], frozenset({PLACE})
                )
                if place_key is not None:
                    add_edge(
                        "HAPPENED_AT",
                        event_key,
                        place_key,
                        f"{event['title']} happened at {event['place']}.",
                        event["verses"],
                        passage_id,
                    )
        for relation in validated["relations"]:
            source_kinds, target_kinds = RELATION_DOMAINS[relation["type"]]
            source = resolve_endpoint(
                resolution, passage_id, relation["source"], source_kinds
            )
            target = resolve_endpoint(
                resolution, passage_id, relation["target"], target_kinds
            )
            if source is not None and target is not None and source != target:
                add_edge(
                    relation["type"],
                    source,
                    target,
                    relation["description"],
                    relation["verses"],
                    passage_id,
                    relation["other_label"],
                )

    def in_verse_order(verse_refs: set[str]) -> list[str]:
        return sorted(verse_refs, key=verse_order.__getitem__)

    finished_entities = []
    for entity in entities.values():
        name = entity["names"].most_common(1)[0][0]
        verses = in_verse_order(entity["mention_verses"])
        finished = {
            "id": entity["id"],
            "kind": entity["kind"],
            "name": name,
            "aliases": sorted((entity["aliases"] | set(entity["names"])) - {name}),
            "description": entity["descriptions"][0] if entity["descriptions"] else "",
            "description_count": len(entity["descriptions"]),
            "mention_verses": verses,
            "mention_count": len(verses),
            "first_verse": verses[0] if verses else None,
            "passages": sorted(entity["passages"]),
        }
        if entity["kind"] == PERSON:
            finished["gender"] = entity["genders"].most_common(1)[0][0]
        if entity["kind"] == PLACE:
            known_kinds = Counter(
                {
                    kind: count
                    for kind, count in entity["place_kinds"].items()
                    if kind != "Unknown"
                }
            )
            finished["place_kind"] = (known_kinds or entity["place_kinds"]).most_common(
                1
            )[0][0]
        finished_entities.append(finished)

    finished_edges = [
        {
            "id": edge["id"],
            "type": edge["type"],
            "source": edge["source"],
            "target": edge["target"],
            "other_label": edge["other_label"],
            "description": edge["descriptions"][0],
            "evidence_verses": in_verse_order(edge["evidence_verses"]),
            "evidence_passages": sorted(edge["evidence_passages"]),
            "support_count": len(edge["evidence_passages"]),
        }
        for edge in edges.values()
    ]
    if layout:
        add_graph_layout(finished_entities, finished_edges)
    return (
        sorted(finished_entities, key=lambda entity: entity["id"]),
        sorted(finished_edges, key=lambda edge: edge["id"]),
    )


def add_graph_layout(entities: list[dict], edges: list[dict]) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(entity["id"] for entity in entities))
    for edge in sorted(edges, key=lambda edge: edge["id"]):
        if graph.has_edge(edge["source"], edge["target"]):
            graph[edge["source"]][edge["target"]]["weight"] += edge["support_count"]
        else:
            graph.add_edge(edge["source"], edge["target"], weight=edge["support_count"])

    communities = nx.community.leiden_communities(
        graph, weight="weight", metric="modularity", seed=GRAPH_SEED
    )
    community_of = {
        node: index
        for index, members in enumerate(
            sorted(communities, key=lambda members: (-len(members), min(members)))
        )
        for node in members
    }
    # Strong gravity keeps unconnected entities next to the core instead of in a far ring.
    positions = nx.forceatlas2_layout(
        graph, weight="weight", max_iter=300, strong_gravity=True, seed=GRAPH_SEED
    )
    for entity in entities:
        x, y = positions[entity["id"]]
        entity["degree"] = graph.degree(entity["id"])
        entity["community"] = community_of[entity["id"]]
        entity["x"] = round(float(x), 4)
        entity["y"] = round(float(y), 4)


INDEX_DIR = DATA_DIR / "index"
