from kgrag.resolve import AliasRules, resolve_v1


def person(
    name: str, verses: list[str], aliases: list[str] | None = None, gender: str = "male"
) -> dict:
    return {
        "name": name,
        "aliases": aliases or [],
        "gender": gender,
        "description": f"{name} in {verses[0]}",
        "named_in": verses,
    }


def father_of(source: str, target: str, verse: str) -> dict:
    return {
        "type": "FATHER_OF",
        "source": source,
        "target": target,
        "description": "",
        "verses": [verse],
        "other_label": None,
    }


def spouse_of(source: str, target: str, verse: str) -> dict:
    return {
        "type": "SPOUSE_OF",
        "source": source,
        "target": target,
        "description": "",
        "verses": [verse],
        "other_label": None,
    }


def extraction(
    passage_id: str, people: list[dict], relations: list[dict] | None = None
) -> dict:
    return {
        "passage_id": passage_id,
        "validated": {
            "people": people,
            "places": [],
            "groups": [],
            "events": [],
            "relations": relations or [],
        },
    }


def never_called(occurrence, cluster):
    raise AssertionError("the confirmer should not be needed")


VERSE_ORDER = {
    ref: index
    for index, ref in enumerate(
        [
            "Gen.4.18",
            "Gen.4.19",
            "Gen.5.25",
            "Gen.5.29",
            "Gen.12.5",
            "Gen.17.5",
            "Gen.17.15",
            "Gen.22.23",
            "Gen.28.5",
            "Gen.29.5",
            "Gen.36.2",
            "Gen.36.4",
        ]
    )
}


def test_two_lamechs_with_different_fathers_stay_separate():
    extractions = [
        extraction(
            "Gen.4.p2",
            [person("Methusael", ["Gen.4.18"]), person("Lamech", ["Gen.4.18"])],
            [father_of("Methusael", "Lamech", "Gen.4.18")],
        ),
        extraction(
            "Gen.5.p2",
            [person("Methuselah", ["Gen.5.25"]), person("Lamech", ["Gen.5.25"])],
            [father_of("Methuselah", "Lamech", "Gen.5.25")],
        ),
    ]

    mapping, decisions = resolve_v1(extractions, VERSE_ORDER, never_called)

    assert mapping[("Gen.4.p2", "person", "Lamech")] == "per_lamech_gen_4_18"
    assert mapping[("Gen.5.p2", "person", "Lamech")] == "per_lamech_gen_5_25"
    assert ("new", "different father") in {
        (d.verdict, d.rule)
        for d in decisions
        if d.occurrence == ("Gen.5.p2", "person", "Lamech")
    }


def test_rename_within_one_passage_merges_abram_and_abraham():
    extractions = [
        extraction(
            "Gen.12.p1",
            [
                person("Abram", ["Gen.12.5"]),
                person("Sarai", ["Gen.12.5"], gender="female"),
            ],
            [spouse_of("Abram", "Sarai", "Gen.12.5")],
        ),
        extraction(
            "Gen.17.p1",
            [
                person("Abraham", ["Gen.17.5"], aliases=["Abram"]),
                person("Sarah", ["Gen.17.15"], aliases=["Sarai"], gender="female"),
            ],
            [spouse_of("Abraham", "Sarah", "Gen.17.15")],
        ),
    ]

    mapping, _ = resolve_v1(extractions, VERSE_ORDER, never_called)

    assert (
        mapping[("Gen.12.p1", "person", "Abram")]
        == mapping[("Gen.17.p1", "person", "Abraham")]
    )
    assert (
        mapping[("Gen.12.p1", "person", "Sarai")]
        == mapping[("Gen.17.p1", "person", "Sarah")]
    )


def test_gender_conflict_blocks_merge():
    extractions = [
        extraction("Gen.4.p2", [person("Adah", ["Gen.4.19"], gender="female")]),
        extraction("Gen.36.p1", [person("Adah", ["Gen.36.2"], gender="male")]),
    ]

    mapping, _ = resolve_v1(extractions, VERSE_ORDER, never_called)

    assert (
        mapping[("Gen.4.p2", "person", "Adah")]
        != mapping[("Gen.36.p1", "person", "Adah")]
    )


def test_aliases_yaml_cannot_link_overrides_name_match():
    extractions = [
        extraction("Gen.4.p2", [person("Lamech", ["Gen.4.18"])]),
        extraction("Gen.5.p2", [person("Lamech", ["Gen.5.25"])]),
    ]

    mapping, decisions = resolve_v1(
        extractions,
        VERSE_ORDER,
        never_called,
        AliasRules(cannot_link=(("lamech@Gen.4.18", "lamech@Gen.5.25"),)),
    )

    assert (
        mapping[("Gen.4.p2", "person", "Lamech")]
        != mapping[("Gen.5.p2", "person", "Lamech")]
    )
    assert decisions[-1].rule == "aliases.yaml cannot_link"


def test_same_name_with_different_relatives_asks_the_confirmer_and_follows_it():
    extractions = [
        extraction(
            "Gen.4.p2",
            [
                person("Lamech", ["Gen.4.19"]),
                person("Adah", ["Gen.4.19"], gender="female"),
            ],
            [spouse_of("Lamech", "Adah", "Gen.4.19")],
        ),
        extraction(
            "Gen.36.p1",
            [
                person("Esau", ["Gen.36.2"]),
                person("Adah", ["Gen.36.2"], gender="female"),
            ],
            [spouse_of("Esau", "Adah", "Gen.36.2")],
        ),
    ]
    asked = []

    def confirm(occurrence, cluster):
        asked.append((occurrence.key, sorted(cluster.names)))
        return False

    mapping, decisions = resolve_v1(extractions, VERSE_ORDER, confirm)

    assert asked == [(("Gen.36.p1", "person", "Adah"), ["adah"])]
    assert mapping[("Gen.4.p2", "person", "Adah")] == "per_adah_gen_4_19"
    assert mapping[("Gen.36.p1", "person", "Adah")] == "per_adah_gen_36_2"
    assert [
        (d.verdict, d.rule)
        for d in decisions
        if d.occurrence == ("Gen.36.p1", "person", "Adah")
    ] == [("new", "LLM said different")]


def test_relatives_are_compared_by_identity_so_a_renamed_father_is_not_a_different_father():
    extractions = [
        extraction(
            "Gen.16.p2",
            [person("Abram", ["Gen.12.5"]), person("Ishmael", ["Gen.12.5"])],
            [father_of("Abram", "Ishmael", "Gen.12.5")],
        ),
        extraction("Gen.17.p1", [person("Abraham", ["Gen.17.5"], aliases=["Abram"])]),
        extraction(
            "Gen.17.p2",
            [person("Abraham", ["Gen.17.15"]), person("Ishmael", ["Gen.17.15"])],
            [father_of("Abraham", "Ishmael", "Gen.17.15")],
        ),
    ]

    mapping, _ = resolve_v1(extractions, VERSE_ORDER, never_called)

    assert (
        mapping[("Gen.16.p2", "person", "Ishmael")]
        == mapping[("Gen.17.p2", "person", "Ishmael")]
        == "per_ishmael"
    )


def test_a_father_one_generation_up_asks_the_confirmer_instead_of_splitting_the_person():
    extractions = [
        extraction(
            "Gen.22.p2",
            [person("Nahor", ["Gen.22.23"]), person("Bethuel", ["Gen.22.23"])],
            [father_of("Nahor", "Bethuel", "Gen.22.23")],
        ),
        extraction(
            "Gen.28.p1",
            [person("Bethuel", ["Gen.28.5"]), person("Laban", ["Gen.28.5"])],
            [father_of("Bethuel", "Laban", "Gen.28.5")],
        ),
        extraction(
            "Gen.29.p1",
            [person("Nahor", ["Gen.29.5"]), person("Laban", ["Gen.29.5"])],
            [father_of("Nahor", "Laban", "Gen.29.5")],
        ),
    ]
    asked = []

    def confirm(occurrence, cluster):
        asked.append(occurrence.key)
        return True

    mapping, _ = resolve_v1(extractions, VERSE_ORDER, confirm)

    assert asked == [
        ("Gen.28.p1", "person", "Bethuel"),
        ("Gen.29.p1", "person", "Laban"),
        ("Gen.29.p1", "person", "Nahor"),
    ]
    assert (
        mapping[("Gen.28.p1", "person", "Laban")]
        == mapping[("Gen.29.p1", "person", "Laban")]
        == "per_laban"
    )
