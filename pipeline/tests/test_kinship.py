from kgrag.build import edge_id
from kgrag.kinship import FOREFATHER, KINSMAN, KinshipFlag, relabel_loose_kinship


def link(edge_type: str, source: str, target: str, verse: str) -> dict:
    return {
        "id": edge_id(edge_type, source, target, None),
        "type": edge_type,
        "source": source,
        "target": target,
        "other_label": None,
        "description": f"{source} {edge_type} {target}",
        "evidence_verses": [verse],
        "evidence_passages": ["p"],
        "support_count": 1,
    }


NAHOR_LABAN = link("FATHER_OF", "per_nahor", "per_laban", "Gen.29.5")
ABRAM_LOT = link("SIBLING_OF", "per_abram", "per_lot", "Gen.14.14")
EDGES = [
    link("FATHER_OF", "per_nahor", "per_bethuel", "Gen.22.23"),
    link("FATHER_OF", "per_bethuel", "per_laban", "Gen.28.5"),
    NAHOR_LABAN,
    link("FATHER_OF", "per_terah", "per_abram", "Gen.11.26"),
    link("FATHER_OF", "per_terah", "per_haran", "Gen.11.27"),
    link("FATHER_OF", "per_haran", "per_lot", "Gen.11.27"),
    ABRAM_LOT,
]


def test_loose_son_and_brother_are_relabelled_as_forefather_and_kinsman():
    flags = []

    def check(flag):
        flags.append(flag)
        return "loose", f"{flag.rule} read loosely"

    edges, decisions = relabel_loose_kinship(EDGES, check)

    assert flags == sorted(
        [
            KinshipFlag(
                NAHOR_LABAN["id"],
                "FATHER_OF",
                "per_nahor",
                "per_laban",
                FOREFATHER,
                "per_bethuel",
                ("Gen.29.5", "Gen.28.5", "Gen.22.23"),
            ),
            KinshipFlag(
                ABRAM_LOT["id"],
                "SIBLING_OF",
                "per_abram",
                "per_lot",
                KINSMAN,
                "per_haran",
                ("Gen.14.14", "Gen.11.27", "Gen.11.26"),
            ),
        ],
        key=lambda flag: flag.edge_id,
    )
    assert [decision["verdict"] for decision in decisions] == ["loose", "loose"]
    relabelled = {
        edge["other_label"]: edge for edge in edges if edge["type"] == "OTHER"
    }
    assert relabelled == {
        "forefather of": {
            **NAHOR_LABAN,
            "id": edge_id("OTHER", "per_nahor", "per_laban", "forefather of"),
            "type": "OTHER",
            "other_label": "forefather of",
            "description": f"{FOREFATHER} read loosely",
        },
        "kinsman of": {
            **ABRAM_LOT,
            "id": edge_id("OTHER", "per_abram", "per_lot", "kinsman of"),
            "type": "OTHER",
            "other_label": "kinsman of",
            "description": f"{KINSMAN} read loosely",
        },
    }
    assert len(edges) == len(EDGES)


def test_links_judged_literal_stay_as_they_are():
    edges, decisions = relabel_loose_kinship(
        EDGES, lambda flag: ("literal", "two people share the name")
    )

    assert edges == sorted(EDGES, key=lambda edge: edge["id"])
    assert [decision["verdict"] for decision in decisions] == ["literal", "literal"]
