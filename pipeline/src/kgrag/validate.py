import re
from dataclasses import dataclass

from kgrag.schema import (
    GROUP,
    PERSON,
    PLACE,
    RELATION_DOMAINS,
    Event,
    PassageExtraction,
    Relation,
)

VERSE_LINE = re.compile(r"^\[(\S+)\] (.*)$")
LEADING_ARTICLE = re.compile(r"^the\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Drop:
    kind: str
    item: str
    reason: str


def verse_texts(passage_text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for line in passage_text.splitlines()
        if (match := VERSE_LINE.match(line))
    }


def name_appears(name: str, text: str) -> bool:
    # Case-sensitive and capitalised on purpose: KJV capitalises names, so this rejects
    # common nouns given as aliases ("the field") and "lord" meaning a human master.
    bare_name = LEADING_ARTICLE.sub("", name.strip()).replace("’", "'")
    if not bare_name or not bare_name[0].isupper():
        return False
    return (
        re.search(rf"(?<!\w){re.escape(bare_name)}(?!\w)", text.replace("’", "'"))
        is not None
    )


def validate_extraction(
    extraction: PassageExtraction, passage_text: str
) -> tuple[PassageExtraction, list[Drop]]:
    verses = verse_texts(passage_text)
    drops: list[Drop] = []
    name_index: dict[str, dict[str, str]] = {PERSON: {}, PLACE: {}, GROUP: {}}

    def keep_named_entities(kind: str, entities: list) -> list:
        kept = []
        for entity in entities:
            if entity.name in name_index[kind]:
                drops.append(Drop(kind, entity.name, "duplicate entity"))
                continue
            names = [entity.name, *entity.aliases]
            named_in = []
            for verse_ref in entity.named_in:
                if verse_ref not in verses:
                    drops.append(
                        Drop(kind, entity.name, f"verse {verse_ref} not in passage")
                    )
                elif any(name_appears(name, verses[verse_ref]) for name in names):
                    named_in.append(verse_ref)
                else:
                    drops.append(Drop(kind, entity.name, f"not named in {verse_ref}"))
            if not named_in:
                drops.append(Drop(kind, entity.name, "never named in passage"))
                continue
            for name in names:
                name_index[kind].setdefault(name, entity.name)
            kept.append(entity.model_copy(update={"named_in": named_in}))
        return kept

    people = keep_named_entities(PERSON, extraction.people)
    places = keep_named_entities(PLACE, extraction.places)
    groups = keep_named_entities(GROUP, extraction.groups)

    def resolve(name: str, allowed_kinds: frozenset[str]) -> str | None:
        for kind in (PERSON, PLACE, GROUP):
            if kind in allowed_kinds and name in name_index[kind]:
                return name_index[kind][name]
        return None

    def verses_in_passage(verse_refs: list[str]) -> list[str]:
        return [verse_ref for verse_ref in verse_refs if verse_ref in verses]

    relations: list[Relation] = []
    for relation in extraction.relations:
        label = f"{relation.type} {relation.source} -> {relation.target}"
        source_kinds, target_kinds = RELATION_DOMAINS[relation.type]
        source = resolve(relation.source, source_kinds)
        target = resolve(relation.target, target_kinds)
        supported_verses = verses_in_passage(relation.verses)
        if relation.type == "OTHER" and not (relation.other_label or "").strip():
            drops.append(Drop("relation", label, "OTHER without other_label"))
        elif source is None:
            drops.append(
                Drop(
                    "relation",
                    label,
                    f"source is not a known {'/'.join(sorted(source_kinds))}",
                )
            )
        elif target is None:
            drops.append(
                Drop(
                    "relation",
                    label,
                    f"target is not a known {'/'.join(sorted(target_kinds))}",
                )
            )
        elif source == target:
            drops.append(
                Drop("relation", label, "source and target are the same entity")
            )
        elif not supported_verses:
            drops.append(Drop("relation", label, "no supporting verse in passage"))
        else:
            relations.append(
                relation.model_copy(
                    update={
                        "source": source,
                        "target": target,
                        "verses": supported_verses,
                        "other_label": relation.other_label
                        if relation.type == "OTHER"
                        else None,
                    }
                )
            )

    events: list[Event] = []
    for event in extraction.events:
        supported_verses = verses_in_passage(event.verses)
        if not supported_verses:
            drops.append(Drop("event", event.title, "no supporting verse in passage"))
            continue
        participants = []
        for participant in event.participants:
            resolved = resolve(participant, frozenset({PERSON, GROUP}))
            if resolved is None:
                drops.append(
                    Drop(
                        "event",
                        event.title,
                        f"participant {participant} is not a known person/group",
                    )
                )
            elif resolved not in participants:
                participants.append(resolved)
        place = resolve(event.place, frozenset({PLACE})) if event.place else None
        if event.place and place is None:
            drops.append(
                Drop("event", event.title, f"place {event.place} is not a known place")
            )
        events.append(
            event.model_copy(
                update={
                    "participants": participants,
                    "place": place,
                    "verses": supported_verses,
                }
            )
        )

    return (
        PassageExtraction(
            people=people,
            places=places,
            groups=groups,
            events=events,
            relations=relations,
        ),
        drops,
    )
