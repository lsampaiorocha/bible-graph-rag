import gzip
import json
from pathlib import Path

import yaml

from kgrag.build import build_index
from kgrag.corpus import REPO_ROOT, read_jsonl
from kgrag.resolve import AliasRules, Cluster, Confirmer, Occurrence, resolve_v1
from kgrag_eval.report import build_report
from kgrag_eval.theographic import gold_from_records, in_scope, parse_md_mentions

FIXTURE_DIR = REPO_ROOT / "eval" / "fixtures" / "genesis"
SCOPE = "Gen"
LLM_RULES = {"LLM confirmed same": True, "LLM said different": False}
KEPT_FIELDS = {
    "verses": ("osisRef", "verseID", "verseText", "people", "places", "mdText"),
    "people": (
        "slug",
        "name",
        "gender",
        "father",
        "mother",
        "partners",
        "siblings",
        "birthPlace",
        "deathPlace",
    ),
    "places": ("slug", "featureType"),
    "events": ("title", "verses", "participants"),
}


def occurrence_key(passage_id: str, kind: str, name: str) -> str:
    return f"{passage_id}|{kind}|{name}"


def replay_confirmer(decisions: dict[str, bool]) -> Confirmer:
    # Answers the resolver's "same entity?" questions from a recorded run, so the
    # benchmark reruns without calling an LLM.
    def confirm(occurrence: Occurrence, cluster: Cluster) -> bool:
        return decisions.get(occurrence_key(*occurrence.key), False)

    return confirm


def write_json_gz(path: Path, value: object) -> None:
    path.write_bytes(
        gzip.compress(json.dumps(value, sort_keys=True).encode("utf-8"), mtime=0)
    )


def read_json_gz(path: Path) -> object:
    return json.loads(gzip.decompress(path.read_bytes()))


def make_fixture(
    extractions: list[dict], audit_path: Path, records: dict[str, dict[str, dict]]
) -> None:
    verses = {
        rid: v for rid, v in records["verses"].items() if in_scope(v["osisRef"], SCOPE)
    }
    events = {
        rid: event
        for rid, event in records["events"].items()
        if any(verse_id in verses for verse_id in event.get("verses", []))
    }
    slug_to_person = {fields["slug"]: rid for rid, fields in records["people"].items()}
    slug_to_place = {fields["slug"]: rid for rid, fields in records["places"].items()}
    linked = [
        span
        for verse in verses.values()
        for span in parse_md_mentions(verse.get("mdText", ""))
    ]
    person_ids = (
        {pid for verse in verses.values() for pid in verse.get("people", [])}
        | {pid for event in events.values() for pid in event.get("participants", [])}
        | {
            slug_to_person[s.slug]
            for s in linked
            if s.kind == "person" and s.slug in slug_to_person
        }
    )
    place_ids = (
        {pid for verse in verses.values() for pid in verse.get("places", [])}
        | {
            slug_to_place[s.slug]
            for s in linked
            if s.kind == "place" and s.slug in slug_to_place
        }
        | {
            place
            for pid in person_ids
            for field in ("birthPlace", "deathPlace")
            for place in records["people"].get(pid, {}).get(field, [])
        }
    )
    subset = {
        "verses": verses,
        "people": {
            rid: records["people"][rid]
            for rid in person_ids
            if rid in records["people"]
        },
        "places": {
            rid: records["places"][rid] for rid in place_ids if rid in records["places"]
        },
        "events": events,
    }
    trimmed = {
        kind: {
            rid: {key: fields[key] for key in KEPT_FIELDS[kind] if key in fields}
            for rid, fields in table.items()
        }
        for kind, table in subset.items()
    }
    decisions = {
        occurrence_key(*row["occurrence"]): LLM_RULES[row["rule"]]
        for row in read_jsonl(audit_path)
        if row["rule"] in LLM_RULES
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    write_json_gz(FIXTURE_DIR / "gold.json.gz", trimmed)
    write_json_gz(
        FIXTURE_DIR / "extractions.json.gz",
        [
            {"passage_id": e["passage_id"], "validated": e["validated"]}
            for e in extractions
        ],
    )
    (FIXTURE_DIR / "confirm_decisions.json").write_text(
        json.dumps(decisions, indent=1, sort_keys=True), encoding="utf-8"
    )
    (FIXTURE_DIR / "expected_report.json").write_text(
        json.dumps(run_fixture(), indent=1), encoding="utf-8"
    )


def run_fixture() -> dict:
    records = read_json_gz(FIXTURE_DIR / "gold.json.gz")
    extractions = read_json_gz(FIXTURE_DIR / "extractions.json.gz")
    decisions = json.loads(
        (FIXTURE_DIR / "confirm_decisions.json").read_text(encoding="utf-8")
    )
    gold = gold_from_records(
        records["verses"], records["people"], records["places"], records["events"]
    )
    verse_order = {
        verse["osisRef"]: position
        for position, verse in enumerate(
            sorted(records["verses"].values(), key=lambda verse: verse["verseID"])
        )
    }
    resolution, _ = resolve_v1(
        extractions, verse_order, replay_confirmer(decisions), AliasRules()
    )
    entities, edges = build_index(extractions, resolution, verse_order, layout=False)
    corrections = (
        yaml.safe_load(
            (REPO_ROOT / "eval" / "gold_corrections.yaml").read_text(encoding="utf-8")
        )
        or {}
    )
    report = build_report(
        "genesis-fixture", SCOPE, entities, edges, gold, corrections, labels_path=None
    )
    return json.loads(json.dumps(report))
