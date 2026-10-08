import re

KIND_PREFIXES = {"person": "per", "place": "pla", "group": "grp", "event": "evt"}

OccurrenceKey = tuple[str, str, str]


def normalize_name(name: str) -> str:
    name = re.sub(r"^the\s+", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def entity_id(kind: str, name: str) -> str:
    return f"{KIND_PREFIXES[kind]}_{normalize_name(name)}"


def entity_occurrences(extraction: dict) -> list[tuple[str, str]]:
    validated = extraction["validated"]
    return [
        *(("person", person["name"]) for person in validated["people"]),
        *(("place", place["name"]) for place in validated["places"]),
        *(("group", group["name"]) for group in validated["groups"]),
        *(("event", event["title"]) for event in validated["events"]),
    ]


def resolve_naive(extractions: list[dict]) -> dict[OccurrenceKey, str]:
    # Deliberately simple baseline: one entity per exact normalised name and kind, so
    # the two Lamechs collapse into one. Entity resolution v1 is measured against it.
    return {
        (extraction["passage_id"], kind, name): entity_id(kind, name)
        for extraction in extractions
        for kind, name in entity_occurrences(extraction)
    }
