from kgrag.schema import Event, PassageExtraction, Person, Place, Relation
from kgrag.validate import Drop, validate_extraction

GENESIS_4_1_TO_2 = (
    "[Gen.4.1] And Adam knew Eve his wife; and she conceived, and bare Cain, and said, "
    "I have gotten a man from the LORD.\n"
    "[Gen.4.2] And she again bare his brother Abel. And Abel was a keeper of sheep, "
    "but Cain was a tiller of the ground."
)


def test_named_in_verse_without_the_name_is_dropped():
    extraction = PassageExtraction(
        people=[
            Person(
                name="Eve",
                aliases=[],
                gender="female",
                description="Eve bore Cain and Abel.",
                named_in=["Gen.4.1", "Gen.4.2"],
            ),
            Person(
                name="God",
                aliases=["the LORD"],
                gender="male",
                description="The LORD gave Eve a son.",
                named_in=["Gen.4.1"],
            ),
            Person(
                name="Seth",
                aliases=[],
                gender="male",
                description="Seth is not in this passage.",
                named_in=["Gen.4.25"],
            ),
        ],
        places=[],
        groups=[],
        events=[],
        relations=[],
    )

    validated, drops = validate_extraction(extraction, GENESIS_4_1_TO_2)

    assert validated.model_dump()["people"] == [
        {
            "name": "Eve",
            "aliases": [],
            "gender": "female",
            "description": "Eve bore Cain and Abel.",
            "named_in": ["Gen.4.1"],
        },
        {
            "name": "God",
            "aliases": ["the LORD"],
            "gender": "male",
            "description": "The LORD gave Eve a son.",
            "named_in": ["Gen.4.1"],
        },
    ]
    assert drops == [
        Drop("person", "Eve", "not named in Gen.4.2"),
        Drop("person", "Seth", "verse Gen.4.25 not in passage"),
        Drop("person", "Seth", "never named in passage"),
    ]


def test_relations_and_events_are_resolved_to_canonical_names_or_dropped():
    adam = Person(
        name="Adam",
        aliases=[],
        gender="male",
        description="Adam knew Eve.",
        named_in=["Gen.4.1"],
    )
    eve = Person(
        name="Eve",
        aliases=[],
        gender="female",
        description="Eve bore Cain.",
        named_in=["Gen.4.1"],
    )
    cain = Person(
        name="Cain",
        aliases=[],
        gender="male",
        description="Cain tilled the ground.",
        named_in=["Gen.4.1", "Gen.4.2"],
    )
    eden = Place(
        name="Eden",
        aliases=[],
        kind="Region",
        description="Not named here.",
        named_in=["Gen.4.1"],
    )
    extraction = PassageExtraction(
        people=[adam, eve, cain],
        places=[eden],
        groups=[],
        events=[
            Event(
                title="Birth of Cain",
                participants=["Eve", "Cain", "Abel"],
                place=None,
                verses=["Gen.4.1", "Gen.4.9"],
            ),
        ],
        relations=[
            Relation(
                type="MOTHER_OF",
                source="Eve",
                target="Cain",
                description="Eve bare Cain.",
                verses=["Gen.4.1"],
                other_label="ignored",
            ),
            Relation(
                type="BORN_IN",
                source="Cain",
                target="Eden",
                description="Cain was born in Eden.",
                verses=["Gen.4.1"],
                other_label=None,
            ),
            Relation(
                type="SPOUSE_OF",
                source="Adam",
                target="Eve",
                description="Adam knew Eve.",
                verses=["Gen.3.20"],
                other_label=None,
            ),
            Relation(
                type="OTHER",
                source="Cain",
                target="Adam",
                description="Cain served Adam.",
                verses=["Gen.4.2"],
                other_label="",
            ),
        ],
    )

    validated, drops = validate_extraction(extraction, GENESIS_4_1_TO_2)

    assert validated.model_dump()["relations"] == [
        {
            "type": "MOTHER_OF",
            "source": "Eve",
            "target": "Cain",
            "description": "Eve bare Cain.",
            "verses": ["Gen.4.1"],
            "other_label": None,
        },
    ]
    assert validated.model_dump()["events"] == [
        {
            "title": "Birth of Cain",
            "participants": ["Eve", "Cain"],
            "place": None,
            "verses": ["Gen.4.1"],
        },
    ]
    assert drops == [
        Drop("place", "Eden", "not named in Gen.4.1"),
        Drop("place", "Eden", "never named in passage"),
        Drop("relation", "BORN_IN Cain -> Eden", "target is not a known place"),
        Drop("relation", "SPOUSE_OF Adam -> Eve", "no supporting verse in passage"),
        Drop("relation", "OTHER Cain -> Adam", "OTHER without other_label"),
        Drop("event", "Birth of Cain", "participant Abel is not a known person/group"),
    ]


def test_lowercase_common_noun_alias_does_not_count_as_a_name():
    extraction = PassageExtraction(
        people=[],
        places=[
            Place(
                name="Field",
                aliases=["the field"],
                kind="Unknown",
                description="Where Cain killed Abel.",
                named_in=["Gen.4.8"],
            )
        ],
        groups=[],
        events=[],
        relations=[],
    )

    validated, drops = validate_extraction(
        extraction,
        "[Gen.4.8] And Cain talked with Abel his brother: and it came to pass, "
        "when they were in the field, that Cain rose up against Abel his brother, and slew him.",
    )

    assert validated.places == []
    assert drops == [
        Drop("place", "Field", "not named in Gen.4.8"),
        Drop("place", "Field", "never named in passage"),
    ]
