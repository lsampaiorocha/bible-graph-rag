import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from kgrag.validate import verse_texts
from kgrag_eval.theographic import GoldMention

POSSESSIVE = re.compile(r"'s$")


@dataclass
class Tally:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def add(self, predicted: set, gold: set) -> None:
        self.true_positives += len(predicted & gold)
        self.false_positives += len(predicted - gold)
        self.false_negatives += len(gold - predicted)

    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    def recall(self) -> float:
        expected = self.true_positives + self.false_negatives
        return self.true_positives / expected if expected else 0.0

    def f1(self) -> float:
        precision, recall = self.precision(), self.recall()
        return (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )


@dataclass
class Accuracy:
    correct: int = 0
    total: int = 0

    def value(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class ConfigScore:
    passages: int = 0
    person_mentions: Tally = field(default_factory=Tally)
    place_mentions: Tally = field(default_factory=Tally)
    parent_links: Tally = field(default_factory=Tally)
    spouse_links: Tally = field(default_factory=Tally)
    sibling_links: Tally = field(default_factory=Tally)
    gender: Accuracy = field(default_factory=Accuracy)
    place_kind: Accuracy = field(default_factory=Accuracy)
    raw_items: int = 0
    dropped_items: int = 0
    costs_usd: list[float] = field(default_factory=list)
    durations_ms: list[int] = field(default_factory=list)
    output_tokens: int = 0
    thinking_tokens: int = 0

    def combined(self) -> float:
        return statistics.fmean(
            [
                self.person_mentions.f1(),
                self.place_mentions.f1(),
                self.parent_links.f1(),
            ]
        )


def normalize_surface(text: str) -> str:
    # The answer key's link text hyphenates some names ("Beth-el") that the KJV text
    # it was built from writes solid ("Bethel").
    text = re.sub(r"^the\s+", "", text.strip(), flags=re.IGNORECASE)
    text = text.replace("’", "'").replace("-", "")
    return POSSESSIVE.sub("", text).lower()


def verse_mentions(
    entity: dict, kind: str, verse_ref: str, spans: list[GoldMention], fallback: str
) -> set[tuple[str, str]]:
    # Credit the mention to whichever answer-key entity the verse names with one of
    # our names, so "Luz, which is Bethel" matches both places the key keeps apart.
    names = {normalize_surface(name) for name in [entity["name"], *entity["aliases"]]}
    matched = {
        span.slug
        for span in spans
        if span.kind == kind and normalize_surface(span.surface) in names
    }
    return {(verse_ref, slug) for slug in matched or {fallback}}


def map_entities_to_gold(
    entities: list[dict], kind: str, gold_spans: dict[str, list[GoldMention]]
) -> dict[str, str]:
    mapping = {}
    for entity in entities:
        names = {
            normalize_surface(name) for name in [entity["name"], *entity["aliases"]]
        }
        votes = Counter(
            span.slug
            for verse_ref in entity["named_in"]
            for span in gold_spans.get(verse_ref, [])
            if span.kind == kind and normalize_surface(span.surface) in names
        )
        if votes:
            mapping[entity["name"]] = votes.most_common(1)[0][0]
    return mapping


def gold_family_links(
    person_slugs: set[str], people_by_slug: dict[str, dict], slug_by_id: dict[str, str]
) -> dict[str, set]:
    links: dict[str, set] = {"parent": set(), "spouse": set(), "sibling": set()}
    for child in person_slugs:
        record = people_by_slug[child]
        for relation, label in (("father", "FATHER_OF"), ("mother", "MOTHER_OF")):
            for parent in (slug_by_id[i] for i in record.get(relation, [])):
                if parent in person_slugs:
                    links["parent"].add((label, parent, child))
        for relation, label in (("partners", "spouse"), ("siblings", "sibling")):
            for other in (slug_by_id[i] for i in record.get(relation, [])):
                if other in person_slugs:
                    links[label].add(tuple(sorted((child, other))))
    return links


def predicted_family_links(
    relations: list[dict], person_map: dict[str, str]
) -> dict[str, set]:
    links: dict[str, set] = {"parent": set(), "spouse": set(), "sibling": set()}
    for relation in relations:
        source = person_map.get(relation["source"], f"?{relation['source']}")
        target = person_map.get(relation["target"], f"?{relation['target']}")
        if relation["type"] in ("FATHER_OF", "MOTHER_OF"):
            links["parent"].add((relation["type"], source, target))
        elif relation["type"] == "SPOUSE_OF":
            links["spouse"].add(tuple(sorted((source, target))))
        elif relation["type"] == "SIBLING_OF":
            links["sibling"].add(tuple(sorted((source, target))))
    return links


def score_passage(
    score: ConfigScore,
    extraction: dict,
    passage_text: str,
    gold_spans: dict[str, list[GoldMention]],
    people_by_slug: dict[str, dict],
    places_by_slug: dict[str, dict],
    slug_by_id: dict[str, str],
) -> None:
    validated = extraction["validated"]
    verses = verse_texts(passage_text)
    passage_spans = {verse_ref: gold_spans.get(verse_ref, []) for verse_ref in verses}

    person_map = map_entities_to_gold(validated["people"], "person", passage_spans)
    place_map = map_entities_to_gold(validated["places"], "place", passage_spans)
    for kind, entities, mapping, tally in (
        ("person", validated["people"], person_map, score.person_mentions),
        ("place", validated["places"], place_map, score.place_mentions),
    ):
        predicted = {
            mention
            for entity in entities
            for verse_ref in entity["named_in"]
            for mention in verse_mentions(
                entity,
                kind,
                verse_ref,
                passage_spans.get(verse_ref, []),
                mapping.get(entity["name"], f"?{entity['name']}"),
            )
        }
        gold = {
            (verse_ref, span.slug)
            for verse_ref, spans in passage_spans.items()
            for span in spans
            if span.kind == kind
        }
        tally.add(predicted, gold)

    for entity in validated["people"]:
        if (slug := person_map.get(entity["name"])) is not None:
            score.gender.total += 1
            score.gender.correct += (
                entity["gender"] == people_by_slug[slug]["gender"].lower()
            )
    for entity in validated["places"]:
        slug = place_map.get(entity["name"])
        if slug is not None and places_by_slug[slug].get("featureType"):
            score.place_kind.total += 1
            score.place_kind.correct += (
                entity["kind"] == places_by_slug[slug]["featureType"]
            )

    gold_person_slugs = {
        span.slug
        for spans in passage_spans.values()
        for span in spans
        if span.kind == "person"
    }
    gold_links = gold_family_links(gold_person_slugs, people_by_slug, slug_by_id)
    predicted_links = predicted_family_links(validated["relations"], person_map)
    score.parent_links.add(predicted_links["parent"], gold_links["parent"])
    score.spouse_links.add(predicted_links["spouse"], gold_links["spouse"])
    score.sibling_links.add(predicted_links["sibling"], gold_links["sibling"])

    raw = extraction["raw"]
    score.passages += 1
    score.raw_items += sum(
        len(raw[key]) for key in ("people", "places", "groups", "events", "relations")
    )
    score.dropped_items += len(extraction["dropped"])
    score.costs_usd.append(extraction["usage"]["cost_usd"])
    score.durations_ms.append(extraction["usage"]["duration_ms"])
    score.output_tokens += extraction["usage"]["output_tokens"]
    score.thinking_tokens += extraction["usage"]["thinking_tokens"]


def load_extraction(extraction_dir: Path, passage_id: str) -> dict | None:
    path = extraction_dir / f"{passage_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
