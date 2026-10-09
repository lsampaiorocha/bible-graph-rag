from kgrag_eval.score_qa import (
    cohen_kappa,
    key_people_shown,
    lineage_gold,
    score_lineage,
    score_trace,
)
from kgrag_eval.theographic import gold_from_records


def test_score_trace_counts_gold_verses_inside_retrieved_passages():
    trace = {
        "query": {"mode": "vector"},
        "chunks": [
            {"id": "Gen.4.p2", "status": "cited"},
            {"id": "Gen.5.p1", "status": "context"},
            {"id": "Gen.6.p1", "status": "candidate"},
        ],
        "answer": {
            "text": "",
            "sentences": [{"text": "Irad begat Mehujael.", "cites": ["C1"]}],
            "invalid_handles": ["C9"],
            "insufficient_context": False,
        },
        "metrics": {
            "usage": {"input_tokens": 900, "output_tokens": 40, "cost_usd": 0.001},
            "timings_ms": {"retrieve": 20, "generate": 980},
        },
    }
    question = {
        "id": "q17",
        "category": "memory_trap",
        "verses": ["Gen.4.18", "Gen.5.3", "Gen.6.1"],
    }
    passage_verses = {
        "Gen.4.p2": ["Gen.4.17", "Gen.4.18"],
        "Gen.5.p1": ["Gen.5.1", "Gen.5.3"],
        "Gen.6.p1": ["Gen.6.1"],
    }

    assert score_trace(trace, question, passage_verses) == {
        "qid": "q17",
        "category": "memory_trap",
        "mode": "vector",
        "answer": "Irad begat Mehujael.",
        "verse_recall": 2 / 3,
        "cited_verse_recall": 1 / 3,
        "context_verses": 4,
        "invalid_citations": 1,
        "citations": 2,
        "input_tokens": 900,
        "output_tokens": 40,
        "cost_usd": 0.001,
        "latency_ms": 1000,
        "failed": False,
    }


def test_cohen_kappa_discounts_agreement_expected_by_chance():
    pairs = [
        ("correct", "correct"),
        ("correct", "wrong"),
        ("wrong", "wrong"),
        ("wrong", "wrong"),
    ]

    assert cohen_kappa(pairs) == 0.5
    assert cohen_kappa([]) is None


def test_lineage_gold_follows_both_parents_and_stays_inside_genesis():
    gold = gold_from_records(
        verses={
            "v1": {"osisRef": "Gen.5.6", "people": ["rA", "rV", "rS", "rN"]},
            "v2": {"osisRef": "Luke.3.38", "people": ["rL"]},
        },
        people={
            "rA": {"slug": "adam", "name": "Adam"},
            "rV": {"slug": "eve", "name": "Eve"},
            "rS": {"slug": "seth", "name": "Seth", "father": ["rA"], "mother": ["rV"]},
            "rN": {"slug": "enos", "name": "Enos", "father": ["rS"]},
            "rL": {"slug": "later", "name": "Later", "father": ["rN"]},
        },
        places={},
        events={},
    )

    assert lineage_gold(gold, "enos", "ancestors", "Gen") == (
        {"seth", "adam", "eve"},
        {("adam", "seth"), ("eve", "seth"), ("seth", "enos")},
    )
    assert lineage_gold(gold, "adam", "descendants", "Gen") == (
        {"seth", "enos"},
        {("adam", "seth"), ("seth", "enos")},
    )


def test_score_lineage_grades_the_tree_the_graph_shows_and_the_names_in_the_answer():
    trace = {
        "query": {"mode": "agent"},
        "nodes": [
            {"id": "per_noah", "status": "cited"},
            {"id": "per_lamech", "status": "context"},
            {"id": "per_methuselah", "status": "cited"},
            {"id": "per_enoch", "status": "candidate"},
            {"id": "per_stranger", "status": "context"},
        ],
        "edges": [
            {
                "type": "FATHER_OF",
                "status": "context",
                "source": "per_lamech",
                "target": "per_noah",
            },
            {
                "type": "FATHER_OF",
                "status": "cited",
                "source": "per_methuselah",
                "target": "per_lamech",
            },
            {
                "type": "SPOUSE_OF",
                "status": "cited",
                "source": "per_lamech",
                "target": "per_stranger",
            },
        ],
    }

    assert score_lineage(
        trace,
        "Noah's father was Lamech, the son of Methuselah.",
        "noah",
        {"lamech", "methuselah", "enoch"},
        {("lamech", "noah"), ("methuselah", "lamech"), ("enoch", "methuselah")},
        {
            "per_noah": "noah",
            "per_lamech": "lamech",
            "per_methuselah": "methuselah",
            "per_enoch": "enoch",
        },
        {"lamech": {"Lamech"}, "methuselah": {"Methuselah"}, "enoch": {"Enoch"}},
    ) == {
        "answer_name_recall": 2 / 3,
        "tree_people_recall": 2 / 3,
        "tree_people_precision": 2 / 3,
        "tree_link_recall": 2 / 3,
        "tree_link_precision": 1.0,
    }


def test_key_people_shown_counts_answer_key_people_in_the_drawn_graph_for_the_agent_only():
    trace = {
        "query": {"mode": "agent"},
        "nodes": [
            {"id": "per_lot", "status": "cited"},
            {"id": "per_abram", "status": "context"},
            {"id": "per_melchizedek", "status": "candidate"},
        ],
    }
    gold_of_entity = {
        "per_lot": "lot_1830",
        "per_abram": "abraham_58",
        "per_melchizedek": "melchisedec_1991",
    }
    people = ["lot_1830", "abraham_58", "melchisedec_1991"]

    assert key_people_shown(trace, people, gold_of_entity) == 2 / 3
    assert (
        key_people_shown({**trace, "query": {"mode": "vector"}}, people, gold_of_entity)
        is None
    )
