from kgrag.build import edge_id
from kgrag.infer import infer_parent_links

ENTITIES = [
    {"id": "per_lamech", "kind": "person", "gender": "male"},
    {"id": "per_adah", "kind": "person", "gender": "female"},
    {"id": "per_jabal", "kind": "person", "gender": "male"},
    {"id": "per_jubal", "kind": "person", "gender": "male"},
]
EDGES = [
    {
        "type": "SPOUSE_OF",
        "source": "per_adah",
        "target": "per_lamech",
        "evidence_verses": ["Gen.4.19"],
        "evidence_passages": ["Gen.4.p2"],
    },
    {
        "type": "MOTHER_OF",
        "source": "per_adah",
        "target": "per_jabal",
        "evidence_verses": ["Gen.4.20"],
        "evidence_passages": ["Gen.4.p2"],
    },
    {
        "type": "SIBLING_OF",
        "source": "per_jabal",
        "target": "per_jubal",
        "evidence_verses": ["Gen.4.21"],
        "evidence_passages": ["Gen.4.p2"],
    },
]
VERSE_ORDER = {"Gen.4.19": 0, "Gen.4.20": 1, "Gen.4.21": 2}


def test_confirmed_father_lets_a_parentless_brother_take_both_parents_in_later_rounds():
    added, decisions = infer_parent_links(
        ENTITIES, EDGES, VERSE_ORDER, lambda proposal: (True, f"{proposal.rule} fits")
    )

    def inferred(edge_type, parent, child, verses, reason):
        return {
            "id": edge_id(edge_type, parent, child, None),
            "type": edge_type,
            "source": parent,
            "target": child,
            "other_label": None,
            "description": reason,
            "evidence_verses": verses,
            "evidence_passages": ["Gen.4.p2"],
            "support_count": 1,
            "inferred": True,
        }

    assert added == [
        inferred(
            "FATHER_OF",
            "per_lamech",
            "per_jabal",
            ["Gen.4.19", "Gen.4.20"],
            "the mother's only husband fits",
        ),
        inferred(
            "MOTHER_OF",
            "per_adah",
            "per_jubal",
            ["Gen.4.20", "Gen.4.21"],
            "a sibling's parent fits",
        ),
        inferred(
            "FATHER_OF",
            "per_lamech",
            "per_jubal",
            ["Gen.4.19", "Gen.4.20", "Gen.4.21"],
            "the mother's only husband fits",
        ),
    ]
    assert [
        (decision["parent"], decision["child"], decision["rule"])
        for decision in decisions
    ] == [
        ("per_lamech", "per_jabal", "the mother's only husband"),
        ("per_adah", "per_jubal", "a sibling's parent"),
        ("per_lamech", "per_jubal", "the mother's only husband"),
    ]


def test_rejected_proposals_add_nothing_and_are_asked_only_once():
    asked = []

    def reject(proposal):
        asked.append(proposal.key)
        return False, "the verses name another father"

    added, decisions = infer_parent_links(ENTITIES, EDGES, VERSE_ORDER, reject)

    assert added == []
    assert asked == ["FATHER_OF|per_lamech|per_jabal", "MOTHER_OF|per_adah|per_jubal"]
    assert [decision["confirmed"] for decision in decisions] == [False, False]


def test_a_mother_with_two_husbands_proposes_no_father():
    second_husband = {
        "type": "SPOUSE_OF",
        "source": "per_adah",
        "target": "per_jubal",
        "evidence_verses": ["Gen.4.21"],
        "evidence_passages": ["Gen.4.p2"],
    }

    added, decisions = infer_parent_links(
        ENTITIES, [*EDGES[:2], second_husband], VERSE_ORDER, lambda proposal: (True, "")
    )

    assert (added, decisions) == ([], [])
