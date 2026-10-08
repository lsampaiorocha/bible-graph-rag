from kgrag_eval.score_qa import cohen_kappa, score_trace


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
