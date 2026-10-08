from typing import Literal

from pydantic import BaseModel, Field

PlaceKind = Literal[
    "City", "Region", "Landmark", "Mountain", "Water", "Valley", "Island", "Unknown"
]

FAMILY_TYPES = ("FATHER_OF", "MOTHER_OF", "SIBLING_OF", "SPOUSE_OF")
NARRATIVE_TYPES = (
    "RULED",
    "SUCCEEDED",
    "ANOINTED",
    "FOUGHT",
    "KILLED",
    "BETRAYED",
    "BLESSED",
    "CURSED",
    "HEALED",
    "DISCIPLE_OF",
    "SERVED",
    "SPOKE_TO",
    "OTHER",
)
EXTRACTED_RELATION_TYPES = (
    *FAMILY_TYPES,
    "MEMBER_OF",
    "BORN_IN",
    "DIED_IN",
    *NARRATIVE_TYPES,
)
# PARTICIPATED_IN and HAPPENED_AT are derived from events at build time, never extracted.
DERIVED_RELATION_TYPES = ("PARTICIPATED_IN", "HAPPENED_AT")
RelationType = Literal[EXTRACTED_RELATION_TYPES]

PERSON, PLACE, GROUP = "person", "place", "group"
RELATION_DOMAINS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    **{name: (frozenset({PERSON}), frozenset({PERSON})) for name in FAMILY_TYPES},
    "MEMBER_OF": (frozenset({PERSON}), frozenset({GROUP})),
    "BORN_IN": (frozenset({PERSON}), frozenset({PLACE})),
    "DIED_IN": (frozenset({PERSON}), frozenset({PLACE})),
    "RULED": (frozenset({PERSON}), frozenset({PLACE, GROUP})),
    **{
        name: (frozenset({PERSON}), frozenset({PERSON}))
        for name in ("SUCCEEDED", "ANOINTED", "BETRAYED", "HEALED", "DISCIPLE_OF")
    },
    **{
        name: (frozenset({PERSON, GROUP}), frozenset({PERSON, GROUP}))
        for name in ("FOUGHT", "KILLED", "BLESSED", "CURSED", "SERVED", "SPOKE_TO")
    },
    "OTHER": (frozenset({PERSON, PLACE, GROUP}), frozenset({PERSON, PLACE, GROUP})),
}


class Person(BaseModel):
    name: str = Field(
        description="The fullest name used for this person in the passage."
    )
    aliases: list[str] = Field(
        description="Other names or titles used for the same person in this passage."
    )
    gender: Literal["male", "female", "unknown"]
    description: str = Field(
        description="One sentence about this person, based only on the passage."
    )
    named_in: list[str] = Field(
        description="Verse references where the person is called by a name or alias."
    )


class Place(BaseModel):
    name: str
    aliases: list[str]
    kind: PlaceKind
    description: str
    named_in: list[str]


class Group(BaseModel):
    name: str = Field(
        description="A tribe, nation, people or sect, e.g. Levites, Philistines, Pharisees."
    )
    aliases: list[str]
    description: str
    named_in: list[str]


class Event(BaseModel):
    title: str = Field(description="A short title such as 'Birth of Obed'.")
    participants: list[str] = Field(
        description="Names of the people or groups taking part."
    )
    place: str | None = Field(
        description="Name of the place where it happened, if stated."
    )
    verses: list[str]


class Relation(BaseModel):
    type: RelationType
    source: str
    target: str
    description: str = Field(
        description="One sentence stating the relation, based only on the passage."
    )
    verses: list[str] = Field(
        description="Verse references that state or support the relation."
    )
    other_label: str | None = Field(
        description="A short verb phrase, required when type is OTHER."
    )


class PassageExtraction(BaseModel):
    people: list[Person]
    places: list[Place]
    groups: list[Group]
    events: list[Event]
    relations: list[Relation]
