from kgrag_eval.entities import (
    AlignedMention,
    ClusterError,
    align_mentions,
    bcubed,
    cluster_errors,
)
from kgrag_eval.theographic import GoldMention


def test_scoreboard_aligns_mentions_and_reports_merge_and_split_errors():
    entities = [
        {
            "id": "per_lamech",
            "kind": "person",
            "name": "Lamech",
            "aliases": [],
            "mention_verses": ["Gen.4.18", "Gen.4.19", "Gen.5.25", "Gen.5.28"],
        },
        {
            "id": "per_abram",
            "kind": "person",
            "name": "Abram",
            "aliases": [],
            "mention_verses": ["Gen.12.1", "Gen.12.4"],
        },
        {
            "id": "per_abraham",
            "kind": "person",
            "name": "Abraham",
            "aliases": ["Abram"],
            "mention_verses": ["Gen.17.5", "Gen.17.9"],
        },
    ]
    gold_spans = {
        "Gen.4.18": [GoldMention("Lamech", "person", "lamech_1808")],
        "Gen.4.19": [GoldMention("Lamech", "person", "lamech_1808")],
        "Gen.5.25": [GoldMention("Lamech", "person", "lamech_1809")],
        "Gen.5.28": [GoldMention("Lamech", "person", "lamech_1809")],
        "Gen.12.1": [GoldMention("Abram", "person", "abraham_58")],
        "Gen.12.4": [GoldMention("Abram", "person", "abraham_58")],
        "Gen.17.5": [
            GoldMention("Abram", "person", "abraham_58"),
            GoldMention("Abraham", "person", "abraham_58"),
        ],
        "Gen.17.9": [GoldMention("Abraham", "person", "abraham_58")],
    }

    mentions = align_mentions(entities, "person", gold_spans)

    assert mentions == [
        AlignedMention("Gen.12.1", "abraham_58", "per_abram"),
        AlignedMention("Gen.12.4", "abraham_58", "per_abram"),
        AlignedMention("Gen.17.5", "abraham_58", "per_abraham"),
        AlignedMention("Gen.17.9", "abraham_58", "per_abraham"),
        AlignedMention("Gen.4.18", "lamech_1808", "per_lamech"),
        AlignedMention("Gen.4.19", "lamech_1808", "per_lamech"),
        AlignedMention("Gen.5.25", "lamech_1809", "per_lamech"),
        AlignedMention("Gen.5.28", "lamech_1809", "per_lamech"),
    ]
    assert bcubed(mentions) == (0.75, 0.75)
    assert cluster_errors(mentions, "entity", "gold") == [
        ClusterError("per_lamech", (("lamech_1808", 2), ("lamech_1809", 2)))
    ]
    assert cluster_errors(mentions, "gold", "entity") == [
        ClusterError("abraham_58", (("per_abraham", 2), ("per_abram", 2)))
    ]
