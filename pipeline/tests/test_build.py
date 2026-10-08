from kgrag.build import build_index, edge_id
from kgrag.resolve import resolve_naive

LAYOUT_FIELDS = {"degree", "community", "x", "y"}


def extraction(
    passage_id: str, people: list[dict], relations: list[dict], events: list[dict]
) -> dict:
    return {
        "passage_id": passage_id,
        "validated": {
            "people": people,
            "places": [],
            "groups": [],
            "events": events,
            "relations": relations,
        },
    }


def person(name: str, named_in: list[str], description: str = "") -> dict:
    return {
        "name": name,
        "aliases": [],
        "gender": "male",
        "description": description,
        "named_in": named_in,
    }


def test_naive_build_merges_same_names_across_passages_and_counts_supporting_passages():
    extractions = [
        extraction(
            "Gen.4.p2",
            [
                person("Lamech", ["Gen.4.18"], "Lamech son of Methusael."),
                person("Jabal", ["Gen.4.20"]),
            ],
            [
                {
                    "type": "FATHER_OF",
                    "source": "Lamech",
                    "target": "Jabal",
                    "description": "Lamech fathered Jabal.",
                    "verses": ["Gen.4.20"],
                    "other_label": None,
                },
            ],
            [
                {
                    "title": "Birth of Jabal",
                    "participants": ["Jabal"],
                    "place": None,
                    "verses": ["Gen.4.20"],
                }
            ],
        ),
        extraction(
            "Gen.5.p2",
            [
                person("Lamech", ["Gen.5.25"], "Lamech son of Methuselah."),
                person("Noah", ["Gen.5.29"]),
                person("Jabal", ["Gen.5.30"]),
            ],
            [
                {
                    "type": "FATHER_OF",
                    "source": "Lamech",
                    "target": "Jabal",
                    "description": "Lamech begat Jabal.",
                    "verses": ["Gen.5.30"],
                    "other_label": None,
                },
                {
                    "type": "SIBLING_OF",
                    "source": "Noah",
                    "target": "Jabal",
                    "description": "Noah and Jabal are brothers.",
                    "verses": ["Gen.5.30"],
                    "other_label": None,
                },
            ],
            [],
        ),
    ]
    verse_order = {
        ref: index
        for index, ref in enumerate(
            ["Gen.4.18", "Gen.4.20", "Gen.5.25", "Gen.5.29", "Gen.5.30"]
        )
    }

    entities, edges = build_index(extractions, resolve_naive(extractions), verse_order)

    assert [
        {key: value for key, value in entity.items() if key not in LAYOUT_FIELDS}
        for entity in entities
    ] == [
        {
            "id": "evt_birth_of_jabal",
            "kind": "event",
            "name": "Birth of Jabal",
            "aliases": [],
            "description": "",
            "description_count": 0,
            "mention_verses": ["Gen.4.20"],
            "mention_count": 1,
            "first_verse": "Gen.4.20",
            "passages": ["Gen.4.p2"],
        },
        {
            "id": "per_jabal",
            "kind": "person",
            "name": "Jabal",
            "aliases": [],
            "description": "",
            "description_count": 0,
            "mention_verses": ["Gen.4.20", "Gen.5.30"],
            "mention_count": 2,
            "first_verse": "Gen.4.20",
            "passages": ["Gen.4.p2", "Gen.5.p2"],
            "gender": "male",
        },
        {
            "id": "per_lamech",
            "kind": "person",
            "name": "Lamech",
            "aliases": [],
            "description": "Lamech son of Methusael.",
            "description_count": 2,
            "mention_verses": ["Gen.4.18", "Gen.5.25"],
            "mention_count": 2,
            "first_verse": "Gen.4.18",
            "passages": ["Gen.4.p2", "Gen.5.p2"],
            "gender": "male",
        },
        {
            "id": "per_noah",
            "kind": "person",
            "name": "Noah",
            "aliases": [],
            "description": "",
            "description_count": 0,
            "mention_verses": ["Gen.5.29"],
            "mention_count": 1,
            "first_verse": "Gen.5.29",
            "passages": ["Gen.5.p2"],
            "gender": "male",
        },
    ]
    assert sorted(edges, key=lambda edge: edge["type"]) == [
        {
            "id": edge_id("FATHER_OF", "per_lamech", "per_jabal", None),
            "type": "FATHER_OF",
            "source": "per_lamech",
            "target": "per_jabal",
            "other_label": None,
            "description": "Lamech fathered Jabal.",
            "evidence_verses": ["Gen.4.20", "Gen.5.30"],
            "evidence_passages": ["Gen.4.p2", "Gen.5.p2"],
            "support_count": 2,
        },
        {
            "id": edge_id("PARTICIPATED_IN", "per_jabal", "evt_birth_of_jabal", None),
            "type": "PARTICIPATED_IN",
            "source": "per_jabal",
            "target": "evt_birth_of_jabal",
            "other_label": None,
            "description": "Jabal took part in Birth of Jabal.",
            "evidence_verses": ["Gen.4.20"],
            "evidence_passages": ["Gen.4.p2"],
            "support_count": 1,
        },
        {
            "id": edge_id("SIBLING_OF", "per_jabal", "per_noah", None),
            "type": "SIBLING_OF",
            "source": "per_jabal",
            "target": "per_noah",
            "other_label": None,
            "description": "Noah and Jabal are brothers.",
            "evidence_verses": ["Gen.5.30"],
            "evidence_passages": ["Gen.5.p2"],
            "support_count": 1,
        },
    ]
    assert all(LAYOUT_FIELDS <= entity.keys() for entity in entities)
