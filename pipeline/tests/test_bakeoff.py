import pytest

from kgrag_eval.bakeoff import ConfigScore, score_passage
from kgrag_eval.theographic import GoldMention


def test_passage_scoring_maps_names_to_the_answer_key_before_comparing():
    passage_text = (
        "[Ruth.4.13] So Boaz took Ruth, and she was his wife: and when he went in unto her, "
        "the LORD gave her conception, and she bare a son.\n"
        "[Ruth.4.17] And the women her neighbours gave it a name, saying, There is a son born "
        "to Naomi; and they called his name Obed: he is the father of Jesse, the father of David."
    )
    gold_spans = {
        "Ruth.4.13": [
            GoldMention("Boaz", "person", "boaz_519"),
            GoldMention("Ruth", "person", "ruth_2450"),
            GoldMention("LORD", "person", "god_1324"),
        ],
        "Ruth.4.17": [
            GoldMention("Naomi", "person", "naomi_2147"),
            GoldMention("Obed", "person", "obed_2228"),
            GoldMention("Jesse", "person", "jesse_903"),
        ],
    }
    people_by_slug = {
        "boaz_519": {
            "gender": "Male",
            "partners": ["id_ruth"],
            "children": ["id_obed"],
        },
        "ruth_2450": {"gender": "Female", "partners": ["id_boaz"]},
        "god_1324": {"gender": "Male"},
        "naomi_2147": {"gender": "Female"},
        "obed_2228": {"gender": "Male", "father": ["id_boaz"], "mother": ["id_ruth"]},
        "jesse_903": {"gender": "Male", "father": ["id_obed"]},
    }
    slug_by_id = {
        "id_boaz": "boaz_519",
        "id_ruth": "ruth_2450",
        "id_obed": "obed_2228",
    }

    def person(name: str, aliases: list[str], gender: str, named_in: list[str]) -> dict:
        return {
            "name": name,
            "aliases": aliases,
            "gender": gender,
            "description": "",
            "named_in": named_in,
        }

    extraction = {
        "raw": {
            "people": [{}] * 6,
            "places": [],
            "groups": [],
            "events": [],
            "relations": [{}] * 3,
        },
        "validated": {
            "people": [
                person("Boaz", [], "male", ["Ruth.4.13"]),
                person("Ruth", [], "female", ["Ruth.4.13"]),
                person("God", ["the LORD"], "female", ["Ruth.4.13"]),
                person("Obed", [], "male", ["Ruth.4.17"]),
                person("Jesse", [], "male", ["Ruth.4.17"]),
                person("David", [], "male", ["Ruth.4.17"]),
            ],
            "places": [],
            "groups": [],
            "events": [],
            "relations": [
                {"type": "FATHER_OF", "source": "Boaz", "target": "Obed"},
                {"type": "FATHER_OF", "source": "Obed", "target": "Jesse"},
                {"type": "SPOUSE_OF", "source": "Ruth", "target": "Boaz"},
            ],
        },
        "dropped": [
            {"kind": "person", "item": "Naomi", "reason": "never named in passage"}
        ],
        "usage": {
            "cost_usd": 0.002,
            "duration_ms": 6000,
            "output_tokens": 900,
            "thinking_tokens": 0,
        },
    }
    score = ConfigScore()

    score_passage(
        score, extraction, passage_text, gold_spans, people_by_slug, {}, slug_by_id
    )

    assert (
        score.person_mentions.true_positives,
        score.person_mentions.false_positives,
        score.person_mentions.false_negatives,
    ) == (5, 1, 1)
    assert (score.gender.correct, score.gender.total) == (4, 5)
    assert (
        score.parent_links.true_positives,
        score.parent_links.false_positives,
        score.parent_links.false_negatives,
    ) == (2, 0, 1)
    assert score.spouse_links.f1() == pytest.approx(1.0)
    assert (score.raw_items, score.dropped_items) == (9, 1)


def test_merged_place_aliases_are_credited_per_verse_and_hyphens_are_ignored():
    passage_text = (
        "[Gen.35.6] So Jacob came to Luz, which is in the land of Canaan, that is, Bethel, "
        "he and all the people that were with him.\n"
        "[Gen.35.9] And God appeared unto Jacob again, when he came out of Padanaram, and blessed him."
    )
    gold_spans = {
        "Gen.35.6": [
            GoldMention("Luz", "place", "luz_737"),
            GoldMention("Beth-el", "place", "bethel_202"),
        ],
        "Gen.35.9": [GoldMention("Padan-aram", "place", "paddan-aram_916")],
    }
    places_by_slug = {
        "luz_737": {"featureType": "City"},
        "bethel_202": {"featureType": "City"},
        "paddan-aram_916": {"featureType": "Region"},
    }

    def place(name: str, aliases: list[str], kind: str, named_in: list[str]) -> dict:
        return {
            "name": name,
            "aliases": aliases,
            "kind": kind,
            "description": "",
            "named_in": named_in,
        }

    extraction = {
        "raw": {
            "people": [],
            "places": [{}, {}],
            "groups": [],
            "events": [],
            "relations": [],
        },
        "validated": {
            "people": [],
            "places": [
                place("Bethel", ["Luz"], "City", ["Gen.35.6"]),
                place("Padanaram", [], "Region", ["Gen.35.9"]),
            ],
            "groups": [],
            "events": [],
            "relations": [],
        },
        "dropped": [],
        "usage": {
            "cost_usd": 0.002,
            "duration_ms": 6000,
            "output_tokens": 900,
            "thinking_tokens": 0,
        },
    }
    score = ConfigScore()

    score_passage(score, extraction, passage_text, gold_spans, {}, places_by_slug, {})

    assert (
        score.place_mentions.true_positives,
        score.place_mentions.false_positives,
        score.place_mentions.false_negatives,
    ) == (3, 0, 0)
    assert (score.place_kind.correct, score.place_kind.total) == (2, 2)
