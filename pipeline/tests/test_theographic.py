from kgrag_eval.theographic import (
    GoldMention,
    GoldStats,
    compute_gold_stats,
    parse_md_mentions,
)


def test_md_text_parser_returns_gold_mentions_for_gen_12_5():
    md_text = (
        "And [Abram]([/person/abraham_58) took [Sarai]([/person/sarah_2473) his wife, and "
        "[Lot]([/person/lot_1830) his brother’s son, and all their substance that they had "
        "gathered, and the souls that they had gotten in [Haran;](/place/haran_527) and they "
        "went forth to go into the land of [Canaan;]([/person/canaan_914) and into the land of "
        "[Canaan]([/person/canaan_914) they came."
    )

    assert parse_md_mentions(md_text) == [
        GoldMention(surface="Abram", kind="person", slug="abraham_58"),
        GoldMention(surface="Sarai", kind="person", slug="sarah_2473"),
        GoldMention(surface="Lot", kind="person", slug="lot_1830"),
        GoldMention(surface="Haran", kind="place", slug="haran_527"),
        GoldMention(surface="Canaan", kind="person", slug="canaan_914"),
        GoldMention(surface="Canaan", kind="person", slug="canaan_914"),
    ]


def test_gold_stats_count_only_relatives_tagged_inside_the_scope():
    verses = {
        "v1": {"osisRef": "Gen.4.1", "people": ["adam", "eve", "cain"], "places": []},
        "v2": {"osisRef": "Gen.4.2", "people": ["abel", "cain"], "places": ["eden"]},
        "v3": {"osisRef": "Exod.1.1", "people": ["moses"], "places": ["egypt"]},
    }
    people = {
        "adam": {"name": "Adam", "partners": ["eve"]},
        "eve": {"name": "Eve", "partners": ["adam"]},
        "cain": {
            "name": "Cain",
            "father": ["adam"],
            "mother": ["eve"],
            "siblings": ["abel", "seth"],
        },
        "abel": {
            "name": "Abel",
            "father": ["adam"],
            "mother": ["eve"],
            "siblings": ["cain"],
        },
        "moses": {"name": "Moses"},
    }

    assert compute_gold_stats(verses, people, scope="Gen") == GoldStats(
        verses=2,
        person_verse_pairs=5,
        people=4,
        place_verse_pairs=1,
        places=1,
        father_pairs=2,
        mother_pairs=2,
        partner_pairs=2,
        sibling_pairs=2,
        shared_name_groups=0,
    )
