import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

KIND_PREFIXES = {"person": "per", "place": "pla", "group": "grp", "event": "evt"}
KIN_RELATIONS = ("fathers", "mothers", "spouses", "children", "siblings")

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


@dataclass
class Occurrence:
    key: OccurrenceKey
    names: set[str]
    gender: str | None
    verses: list[str]
    description: str
    chapter: tuple[str, int]
    kin: dict[str, set[str]] = field(
        default_factory=lambda: {relation: set() for relation in KIN_RELATIONS}
    )

    @property
    def passage_id(self) -> str:
        return self.key[0]

    @property
    def kind(self) -> str:
        return self.key[1]

    def refs(self) -> set[str]:
        return {f"{name}@{verse}" for name in self.names for verse in self.verses}

    def has_kin(self) -> bool:
        return any(self.kin.values())


@dataclass
class Cluster:
    kind: str
    occurrences: list[Occurrence] = field(default_factory=list)

    @property
    def names(self) -> set[str]:
        return set().union(*(occurrence.names for occurrence in self.occurrences))

    def kin(self, relation: str) -> set[str]:
        return set().union(
            *(occurrence.kin[relation] for occurrence in self.occurrences)
        )

    def genders(self) -> set[str]:
        return {
            o.gender for o in self.occurrences if o.gender and o.gender != "unknown"
        }

    def has_kin(self) -> bool:
        return any(self.kin(relation) for relation in KIN_RELATIONS)


@dataclass(frozen=True)
class Decision:
    occurrence: OccurrenceKey
    verdict: str
    rule: str
    cluster_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasRules:
    must_link: tuple[tuple[str, str], ...] = ()
    cannot_link: tuple[tuple[str, str], ...] = ()


NO_ALIAS_RULES = AliasRules()
Confirmer = Callable[[Occurrence, Cluster], bool]


def name_key(name: str) -> str:
    name = re.sub(r"^the\s+", "", name.strip().lower()).replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", re.sub(r"['\-]", "", name)).split())


def chapter_of(verse_ref: str) -> tuple[str, int]:
    book, chapter, _ = verse_ref.split(".")
    return book, int(chapter)


def build_occurrences(extractions: list[dict]) -> list[Occurrence]:
    occurrences = []
    for extraction in extractions:
        passage_id = extraction["passage_id"]
        validated = extraction["validated"]
        local: dict[tuple[str, str], Occurrence] = {}
        for kind, records in (
            ("person", validated["people"]),
            ("place", validated["places"]),
            ("group", validated["groups"]),
        ):
            for record in records:
                occurrence = Occurrence(
                    key=(passage_id, kind, record["name"]),
                    names={
                        name_key(name)
                        for name in [record["name"], *record["aliases"]]
                        if name_key(name)
                    },
                    gender=record.get("gender"),
                    verses=record["named_in"],
                    description=record["description"],
                    chapter=chapter_of(record["named_in"][0]),
                )
                local[(kind, record["name"])] = occurrence
                occurrences.append(occurrence)
        people = {
            name: occurrence
            for (kind, name), occurrence in local.items()
            if kind == "person"
        }
        for relation in validated["relations"]:
            source, target = (
                people.get(relation["source"]),
                people.get(relation["target"]),
            )
            if source is None or target is None:
                continue
            if relation["type"] in ("FATHER_OF", "MOTHER_OF"):
                target.kin[
                    "fathers" if relation["type"] == "FATHER_OF" else "mothers"
                ] |= source.names
                source.kin["children"] |= target.names
            elif relation["type"] in ("SPOUSE_OF", "SIBLING_OF"):
                group = "spouses" if relation["type"] == "SPOUSE_OF" else "siblings"
                source.kin[group] |= target.names
                target.kin[group] |= source.names
    return occurrences


class NameClasses:
    """Names learned to belong to one person, such as Abram and Abraham."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, name: str) -> str:
        self.parent.setdefault(name, name)
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def union(self, names: set[str]) -> None:
        roots = sorted({self.find(name) for name in names})
        for root in roots[1:]:
            self.parent[root] = roots[0]

    def canonical(self, names: set[str]) -> set[str]:
        return {self.find(name) for name in names}


def compare(
    occurrence: Occurrence,
    cluster: Cluster,
    rules: AliasRules,
    canonical: Callable[[set[str]], set[str]],
) -> tuple[str, str]:
    refs = occurrence.refs()
    cluster_refs = set().union(*(other.refs() for other in cluster.occurrences))
    for first, second in rules.cannot_link:
        if (first in refs and second in cluster_refs) or (
            second in refs and first in cluster_refs
        ):
            return "cannot", "aliases.yaml cannot_link"
    for first, second in rules.must_link:
        if (first in refs and second in cluster_refs) or (
            second in refs and first in cluster_refs
        ):
            return "must", "aliases.yaml must_link"
    if any(other.passage_id == occurrence.passage_id for other in cluster.occurrences):
        return "cannot", "same passage, extracted as different entities"
    genders = cluster.genders()
    if (
        occurrence.gender
        and occurrence.gender != "unknown"
        and genders
        and occurrence.gender not in genders
    ):
        return "cannot", "gender conflict"
    # Relatives are compared by identity, not spelling: Ishmael's father is "Abram" in
    # Gen 16 and "Abraham" in Gen 17, which must not read as two different fathers.
    kin = {relation: canonical(occurrence.kin[relation]) for relation in KIN_RELATIONS}
    cluster_kin = {
        relation: canonical(cluster.kin(relation)) for relation in KIN_RELATIONS
    }
    for parent in ("fathers", "mothers"):
        if (
            kin[parent]
            and cluster_kin[parent]
            and not kin[parent] & cluster_kin[parent]
        ):
            return "cannot", f"different {parent[:-1]}"
    shared = [
        relation for relation in KIN_RELATIONS if kin[relation] & cluster_kin[relation]
    ]
    if shared:
        return "must", f"shared {', '.join(shared)}"
    if any(other.chapter == occurrence.chapter for other in cluster.occurrences):
        return "must", "same chapter"
    return "uncertain", "same name, no deciding evidence"


def resolve_v1(
    extractions: list[dict],
    verse_order: dict[str, int],
    confirm: Confirmer,
    rules: AliasRules = NO_ALIAS_RULES,
) -> tuple[dict[OccurrenceKey, str], list[Decision]]:
    occurrences = sorted(
        build_occurrences(extractions),
        key=lambda o: (verse_order[o.verses[0]], o.kind, sorted(o.names)),
    )
    clusters: list[Cluster] = []
    decisions: list[Decision] = []
    person_names = NameClasses()
    for occurrence in occurrences:
        if occurrence.kind == "person":
            person_names.union(occurrence.names)
        candidates = [
            c
            for c in clusters
            if c.kind == occurrence.kind and c.names & occurrence.names
        ]
        verdicts = [
            (cluster, *compare(occurrence, cluster, rules, person_names.canonical))
            for cluster in candidates
        ]
        musts = [
            (cluster, rule) for cluster, verdict, rule in verdicts if verdict == "must"
        ]
        uncertain = [
            cluster for cluster, verdict, _ in verdicts if verdict == "uncertain"
        ]
        chosen, rule = None, ""
        if musts:
            chosen, rule = max(musts, key=lambda item: len(item[0].occurrences))
        elif len(uncertain) == 1 and not (
            occurrence.has_kin() and uncertain[0].has_kin()
        ):
            # One possible match and nothing that tells them apart: the common case of the
            # same well-known figure appearing again (God, Joseph), so join without asking.
            chosen, rule = uncertain[0], "only candidate, nothing conflicts"
        else:
            for cluster in uncertain:
                if confirm(occurrence, cluster):
                    chosen, rule = cluster, "LLM confirmed same"
                    break
            else:
                rule = (
                    "LLM said different"
                    if uncertain
                    else (verdicts[0][2] if verdicts else "new name")
                )
        if chosen is None:
            chosen = Cluster(kind=occurrence.kind)
            clusters.append(chosen)
            decisions.append(Decision(occurrence.key, "new", rule))
        else:
            decisions.append(
                Decision(occurrence.key, "joined", rule, tuple(sorted(chosen.names)))
            )
        chosen.occurrences.append(occurrence)
        if occurrence.kind == "person":
            person_names.union(chosen.names)

    mapping = cluster_ids(clusters, verse_order)
    for extraction in extractions:
        for event in extraction["validated"]["events"]:
            mapping[(extraction["passage_id"], "event", event["title"])] = entity_id(
                "event", event["title"]
            )
    return mapping, decisions


def cluster_ids(
    clusters: list[Cluster], verse_order: dict[str, int]
) -> dict[OccurrenceKey, str]:
    def canonical(cluster: Cluster) -> str:
        names = Counter(occurrence.key[2] for occurrence in cluster.occurrences)
        return min(names.items(), key=lambda item: (-item[1], item[0]))[0]

    def first_verse(cluster: Cluster) -> str:
        return min(
            (v for o in cluster.occurrences for v in o.verses),
            key=verse_order.__getitem__,
        )

    base_counts = Counter(
        (cluster.kind, entity_id(cluster.kind, canonical(cluster)))
        for cluster in clusters
    )
    mapping = {}
    for cluster in clusters:
        base = entity_id(cluster.kind, canonical(cluster))
        if base_counts[(cluster.kind, base)] > 1:
            # Namesakes get the verse that first names them: per_lamech_gen_4_18.
            base = f"{base}_{normalize_name(first_verse(cluster))}"
        for occurrence in cluster.occurrences:
            mapping[occurrence.key] = base
    return mapping
