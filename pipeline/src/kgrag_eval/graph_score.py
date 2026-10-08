from collections import defaultdict
from dataclasses import dataclass, field

from kgrag_eval.bakeoff import Tally
from kgrag_eval.entities import align_mentions, majority
from kgrag_eval.theographic import Gold, in_scope

EVENT_MATCH_MIN_JACCARD = 0.25
EVENT_SIZE_BUCKETS = (
    ("1-3 verses", 1, 3),
    ("4-10 verses", 4, 10),
    ("over 10 verses", 11, 10**6),
)
UNDIRECTED = {"SIBLING_OF", "SPOUSE_OF"}


@dataclass
class EventScore:
    gold_events: int = 0
    matched: int = 0
    by_size: dict[str, tuple[int, int]] = field(default_factory=dict)
    participants: Tally = field(default_factory=Tally)

    def recall(self) -> float:
        return self.matched / self.gold_events if self.gold_events else 0.0


@dataclass
class GraphScore:
    triples: dict[str, Tally]
    sibling_closure: Tally
    places: dict[str, Tally]
    events: EventScore


def link_key(edge_type: str, source: str, target: str) -> tuple[str, str, str]:
    if edge_type in UNDIRECTED:
        source, target = sorted((source, target))
    return edge_type, source, target


def scope_people(gold: Gold, scope: str) -> set[str]:
    return {
        gold.person_slug_by_id[record_id]
        for verse in gold.verses.values()
        if in_scope(verse["osisRef"], scope)
        for record_id in verse.get("people", [])
    }


def gold_family_triples(gold: Gold, scope: str) -> dict[str, set[tuple[str, str, str]]]:
    people_in_scope = scope_people(gold, scope)
    triples: dict[str, set] = {
        name: set() for name in ("FATHER_OF", "MOTHER_OF", "SPOUSE_OF", "SIBLING_OF")
    }
    for slug in people_in_scope:
        record = gold.people_by_slug[slug]
        for field_name, edge_type, parent_first in (
            ("father", "FATHER_OF", True),
            ("mother", "MOTHER_OF", True),
            ("partners", "SPOUSE_OF", False),
            ("siblings", "SIBLING_OF", False),
        ):
            for relative_id in record.get(field_name, []):
                relative = gold.person_slug_by_id.get(relative_id)
                if relative in people_in_scope:
                    source, target = (
                        (relative, slug) if parent_first else (slug, relative)
                    )
                    triples[edge_type].add(link_key(edge_type, source, target))
    return triples


def score_graph(
    entities: list[dict], edges: list[dict], gold: Gold, scope: str
) -> GraphScore:
    mentions = align_mentions(entities, "person", gold.spans) + align_mentions(
        entities, "place", gold.spans
    )
    gold_of = majority(mentions, "entity", "gold")

    def mapped(entity_id: str) -> str:
        # An entity the answer key cannot place gets a unique stand-in, so its links can
        # only ever count as wrong.
        return gold_of.get(entity_id, f"?{entity_id}")

    gold_triples = gold_family_triples(gold, scope)
    predicted: dict[str, set] = {name: set() for name in gold_triples}
    for edge in edges:
        if edge["type"] in predicted:
            predicted[edge["type"]].add(
                link_key(edge["type"], mapped(edge["source"]), mapped(edge["target"]))
            )
    triples = {}
    for edge_type in gold_triples:
        tally = Tally()
        tally.add(predicted[edge_type], gold_triples[edge_type])
        triples[edge_type] = tally

    # Siblings are rarely stated outright; children of one parent count as siblings here.
    children_of: dict[str, set[str]] = defaultdict(set)
    for _, parent, child in predicted["FATHER_OF"] | predicted["MOTHER_OF"]:
        children_of[parent].add(child)
    closure = set(predicted["SIBLING_OF"])
    for children in children_of.values():
        closure |= {
            link_key("SIBLING_OF", a, b) for a in children for b in children if a < b
        }
    sibling_closure = Tally()
    sibling_closure.add(closure, gold_triples["SIBLING_OF"])

    places = {}
    for edge_type, field_name in (("BORN_IN", "birthPlace"), ("DIED_IN", "deathPlace")):
        gold_links = {
            (edge_type, slug, gold.place_slug_by_id[place_id])
            for slug in scope_people(gold, scope)
            for place_id in gold.people_by_slug[slug].get(field_name, [])
            if place_id in gold.place_slug_by_id
        }
        predicted_links = {
            (edge_type, mapped(edge["source"]), mapped(edge["target"]))
            for edge in edges
            if edge["type"] == edge_type
        }
        tally = Tally()
        tally.add(predicted_links, gold_links)
        places[edge_type] = tally

    return GraphScore(
        triples,
        sibling_closure,
        places,
        score_events(entities, edges, gold, scope, mapped),
    )


def score_events(
    entities: list[dict], edges: list[dict], gold: Gold, scope: str, mapped
) -> EventScore:
    verse_ref = {
        record_id: fields["osisRef"] for record_id, fields in gold.verses.items()
    }
    gold_events = [
        event
        for event in gold.events.values()
        if any(
            in_scope(verse_ref.get(verse_id, ""), scope)
            for verse_id in event.get("verses", [])
        )
    ]
    ours = [entity for entity in entities if entity["kind"] == "event"]
    participants_of: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge["type"] == "PARTICIPATED_IN":
            participants_of[edge["target"]].add(mapped(edge["source"]))

    pairs = []
    for gold_index, event in enumerate(gold_events):
        gold_verses = {verse_ref[v] for v in event.get("verses", []) if v in verse_ref}
        for entity in ours:
            our_verses = set(entity["mention_verses"])
            jaccard = (
                len(gold_verses & our_verses) / len(gold_verses | our_verses)
                if gold_verses | our_verses
                else 0.0
            )
            if jaccard >= EVENT_MATCH_MIN_JACCARD:
                pairs.append((-jaccard, gold_index, entity["id"]))
    # Greedy one-to-one matching, best overlap first, ties broken by position and id.
    matched_gold: dict[int, str] = {}
    used: set[str] = set()
    for _, gold_index, entity_id in sorted(pairs):
        if gold_index not in matched_gold and entity_id not in used:
            matched_gold[gold_index] = entity_id
            used.add(entity_id)

    score = EventScore(gold_events=len(gold_events), matched=len(matched_gold))
    for label, low, high in EVENT_SIZE_BUCKETS:
        in_bucket = [
            i
            for i, event in enumerate(gold_events)
            if low <= len(event.get("verses", [])) <= high
        ]
        score.by_size[label] = (
            sum(1 for i in in_bucket if i in matched_gold),
            len(in_bucket),
        )
    for gold_index, entity_id in matched_gold.items():
        gold_participants = {
            gold.person_slug_by_id[p]
            for p in gold_events[gold_index].get("participants", [])
            if p in gold.person_slug_by_id
        }
        score.participants.add(participants_of[entity_id], gold_participants)
    return score
