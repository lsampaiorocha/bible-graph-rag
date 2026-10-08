import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from kgrag_eval.bakeoff import Accuracy, Tally, normalize_surface
from kgrag_eval.theographic import GoldMention

KEEP_APART = [
    ("lamech_1808", "lamech_1809"),
    ("enoch_1191", "enoch_1192"),
    ("nahor_2142", "nahor_2143"),
]
KEEP_TOGETHER = ["abraham_58", "sarah_2473", "israel_682"]
TOGETHER_SHARE = 0.9


@dataclass(frozen=True, order=True)
class AlignedMention:
    verse: str
    gold: str
    entity: str


@dataclass(frozen=True)
class ClusterError:
    key: str
    parts: tuple[tuple[str, int], ...]


@dataclass
class EntityScore:
    mentions: int
    bcubed_precision: float
    bcubed_recall: float
    merge_errors: list[ClusterError]
    split_errors: list[ClusterError]
    mention_f1: dict[str, Tally]
    gender: Accuracy
    place_kind: Accuracy
    hard_cases: list[tuple[str, bool, str]]

    def bcubed_f1(self) -> float:
        precision, recall = self.bcubed_precision, self.bcubed_recall
        return (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )


def align_mentions(
    entities: list[dict], kind: str, gold_spans: dict[str, list[GoldMention]]
) -> list[AlignedMention]:
    # A mention is aligned when the answer key links, in the same verse, a name that is
    # one of the entity's own names; that tells us which real person or place it meant.
    aligned = set()
    for entity in entities:
        if entity["kind"] != kind:
            continue
        names = {
            normalize_surface(name) for name in [entity["name"], *entity["aliases"]]
        }
        for verse in entity["mention_verses"]:
            for span in gold_spans.get(verse, []):
                if span.kind == kind and normalize_surface(span.surface) in names:
                    aligned.add(AlignedMention(verse, span.slug, entity["id"]))
    return sorted(aligned)


def bcubed(mentions: list[AlignedMention]) -> tuple[float, float]:
    if not mentions:
        return 0.0, 0.0
    by_gold: dict[str, set[int]] = defaultdict(set)
    by_entity: dict[str, set[int]] = defaultdict(set)
    for index, mention in enumerate(mentions):
        by_gold[mention.gold].add(index)
        by_entity[mention.entity].add(index)
    precision = statistics.fmean(
        len(by_entity[m.entity] & by_gold[m.gold]) / len(by_entity[m.entity])
        for m in mentions
    )
    recall = statistics.fmean(
        len(by_entity[m.entity] & by_gold[m.gold]) / len(by_gold[m.gold])
        for m in mentions
    )
    return precision, recall


def cluster_errors(
    mentions: list[AlignedMention], group_by: str, part_of: str
) -> list[ClusterError]:
    # A group counts as wrongly merged (or split) when its second-largest part holds at
    # least max(2, 20%) of its mentions, so one stray alias match does not count.
    parts: dict[str, Counter] = defaultdict(Counter)
    for mention in mentions:
        parts[getattr(mention, group_by)][getattr(mention, part_of)] += 1
    errors = []
    for key, counts in sorted(parts.items()):
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) > 1 and ranked[1][1] >= max(2, 0.2 * sum(counts.values())):
            errors.append(ClusterError(key, tuple(ranked)))
    return sorted(errors, key=lambda error: -sum(count for _, count in error.parts))


def majority(
    mentions: list[AlignedMention], group_by: str, pick: str
) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for mention in mentions:
        counts[getattr(mention, group_by)][getattr(mention, pick)] += 1
    return {
        key: min(counter.items(), key=lambda item: (-item[1], item[0]))[0]
        for key, counter in counts.items()
    }


def hard_case_results(
    person_mentions: list[AlignedMention],
) -> list[tuple[str, bool, str]]:
    entity_of_gold = majority(person_mentions, "gold", "entity")
    results = []
    for first, second in KEEP_APART:
        first_entity, second_entity = (
            entity_of_gold.get(first),
            entity_of_gold.get(second),
        )
        passed = (
            first_entity is not None
            and second_entity is not None
            and first_entity != second_entity
        )
        results.append(
            (
                f"keep {first} and {second} apart",
                passed,
                f"{first_entity} / {second_entity}",
            )
        )
    for gold in KEEP_TOGETHER:
        counts = Counter(m.entity for m in person_mentions if m.gold == gold)
        total = sum(counts.values())
        top_entity, top_count = (
            min(counts.items(), key=lambda item: (-item[1], item[0]))
            if counts
            else (None, 0)
        )
        passed = total > 0 and top_count / total >= TOGETHER_SHARE
        detail = ", ".join(
            f"{entity} {count}" for entity, count in counts.most_common(3)
        )
        results.append(
            (f"merge all names of {gold}", passed, detail or "no aligned mentions")
        )
    return results


def score_entities(
    entities: list[dict],
    gold_spans: dict[str, list[GoldMention]],
    gold_tags: dict[str, set[tuple[str, str]]],
    people_by_slug: dict[str, dict],
    places_by_slug: dict[str, dict],
) -> EntityScore:
    person_mentions = align_mentions(entities, "person", gold_spans)
    place_mentions = align_mentions(entities, "place", gold_spans)
    all_mentions = person_mentions + place_mentions
    precision, recall = bcubed(all_mentions)
    gold_of_entity = majority(all_mentions, "entity", "gold")

    mention_f1 = {}
    for kind in ("person", "place"):
        predicted = {
            (verse, gold_of_entity.get(entity["id"], f"?{entity['id']}"))
            for entity in entities
            if entity["kind"] == kind
            for verse in entity["mention_verses"]
        }
        tally = Tally()
        tally.add(predicted, gold_tags[kind])
        mention_f1[kind] = tally

    gender, place_kind = Accuracy(), Accuracy()
    for entity in entities:
        slug = gold_of_entity.get(entity["id"])
        if slug is None:
            continue
        if entity["kind"] == "person" and slug in people_by_slug:
            gender.total += 1
            gender.correct += (
                entity.get("gender") == people_by_slug[slug]["gender"].lower()
            )
        if entity["kind"] == "place" and places_by_slug.get(slug, {}).get(
            "featureType"
        ):
            place_kind.total += 1
            place_kind.correct += (
                entity.get("place_kind") == places_by_slug[slug]["featureType"]
            )

    return EntityScore(
        mentions=len(all_mentions),
        bcubed_precision=precision,
        bcubed_recall=recall,
        merge_errors=cluster_errors(all_mentions, "entity", "gold"),
        split_errors=cluster_errors(all_mentions, "gold", "entity"),
        mention_f1=mention_f1,
        gender=gender,
        place_kind=place_kind,
        hard_cases=hard_case_results(person_mentions),
    )
