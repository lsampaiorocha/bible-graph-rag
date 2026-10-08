import json
import re
from collections import Counter
from dataclasses import dataclass

from kgrag.corpus import fetch_theographic_file

ALL_BOOKS = "Bible"
MENTION_LINK = re.compile(r"\[([^\]]+)\]\(\[?/(person|place)/([^)\s]+)\)")


@dataclass(frozen=True)
class GoldMention:
    surface: str
    kind: str
    slug: str


@dataclass(frozen=True)
class GoldStats:
    verses: int
    person_verse_pairs: int
    people: int
    place_verse_pairs: int
    places: int
    father_pairs: int
    mother_pairs: int
    partner_pairs: int
    sibling_pairs: int
    shared_name_groups: int


def load_records(name: str) -> dict[str, dict]:
    records = json.loads(fetch_theographic_file(name).read_text(encoding="utf-8"))
    return {record["id"]: record["fields"] for record in records}


def parse_md_mentions(md_text: str) -> list[GoldMention]:
    # mdText wraps trailing punctuation inside the link text ("[Haran;]") and mixes
    # "([/person/..." with "(/place/...", so both are normalised here.
    return [
        GoldMention(
            surface=match.group(1).strip(",;:.!?() "),
            kind=match.group(2),
            slug=match.group(3),
        )
        for match in MENTION_LINK.finditer(md_text)
    ]


def in_scope(osis_ref: str, scope: str) -> bool:
    return scope == ALL_BOOKS or osis_ref.split(".")[0] == scope


def compute_gold_stats(
    verses: dict[str, dict], people: dict[str, dict], scope: str
) -> GoldStats:
    scoped_verses = [
        verse for verse in verses.values() if in_scope(verse["osisRef"], scope)
    ]
    people_in_scope = {
        person_id for verse in scoped_verses for person_id in verse.get("people", [])
    }
    places_in_scope = {
        place_id for verse in scoped_verses for place_id in verse.get("places", [])
    }

    def pairs_within_scope(relation: str) -> int:
        return sum(
            1
            for person_id in people_in_scope
            for relative_id in people[person_id].get(relation, [])
            if relative_id in people_in_scope
        )

    name_counts = Counter(people[person_id]["name"] for person_id in people_in_scope)
    return GoldStats(
        verses=len(scoped_verses),
        person_verse_pairs=sum(len(verse.get("people", [])) for verse in scoped_verses),
        people=len(people_in_scope),
        place_verse_pairs=sum(len(verse.get("places", [])) for verse in scoped_verses),
        places=len(places_in_scope),
        father_pairs=pairs_within_scope("father"),
        mother_pairs=pairs_within_scope("mother"),
        partner_pairs=pairs_within_scope("partners"),
        sibling_pairs=pairs_within_scope("siblings"),
        shared_name_groups=sum(1 for count in name_counts.values() if count > 1),
    )


@dataclass(frozen=True)
class Gold:
    verses: dict[str, dict]
    people_by_slug: dict[str, dict]
    places_by_slug: dict[str, dict]
    person_slug_by_id: dict[str, str]
    place_slug_by_id: dict[str, str]
    spans: dict[str, list[GoldMention]]
    dangling_links: int

    def tags(self, scope: str) -> dict[str, set[tuple[str, str]]]:
        tags: dict[str, set[tuple[str, str]]] = {"person": set(), "place": set()}
        for verse in self.verses.values():
            if not in_scope(verse["osisRef"], scope):
                continue
            for record_id in verse.get("people", []):
                tags["person"].add(
                    (verse["osisRef"], self.person_slug_by_id[record_id])
                )
            for record_id in verse.get("places", []):
                tags["place"].add((verse["osisRef"], self.place_slug_by_id[record_id]))
        return tags


def load_gold() -> Gold:
    verses = load_records("verses.json")
    people = load_records("people.json")
    places = load_records("places.json")
    people_by_slug = {fields["slug"]: fields for fields in people.values()}
    places_by_slug = {fields["slug"]: fields for fields in places.values()}
    known_slugs = {"person": people_by_slug, "place": places_by_slug}
    spans = {}
    dangling_links = 0
    for fields in verses.values():
        verse_spans = parse_md_mentions(fields.get("mdText", ""))
        spans[fields["osisRef"]] = [
            span for span in verse_spans if span.slug in known_slugs[span.kind]
        ]
        dangling_links += len(verse_spans) - len(spans[fields["osisRef"]])
    return Gold(
        verses=verses,
        people_by_slug=people_by_slug,
        places_by_slug=places_by_slug,
        person_slug_by_id={
            record_id: fields["slug"] for record_id, fields in people.items()
        },
        place_slug_by_id={
            record_id: fields["slug"] for record_id, fields in places.items()
        },
        spans=spans,
        dangling_links=dangling_links,
    )
