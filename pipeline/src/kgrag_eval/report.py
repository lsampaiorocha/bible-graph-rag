from pathlib import Path

from kgrag_eval.bakeoff import Tally
from kgrag_eval.entities import EntityScore, score_entities
from kgrag_eval.graph_score import GraphScore, score_graph
from kgrag_eval.narrative import read_labels, score_labels
from kgrag_eval.theographic import Gold, apply_corrections


def tally_summary(tally: Tally) -> dict:
    return {
        "precision": round(tally.precision(), 4),
        "recall": round(tally.recall(), 4),
        "f1": round(tally.f1(), 4),
        "gold": tally.true_positives + tally.false_negatives,
    }


def entity_summary(index_id: str, score: EntityScore) -> dict:
    return {
        "index": index_id,
        "aligned_mentions": score.mentions,
        "bcubed_precision": score.bcubed_precision,
        "bcubed_recall": score.bcubed_recall,
        "bcubed_f1": score.bcubed_f1(),
        "merge_errors": len(score.merge_errors),
        "split_errors": len(score.split_errors),
        "person_mention_f1": score.mention_f1["person"].f1(),
        "place_mention_f1": score.mention_f1["place"].f1(),
        "gender_accuracy": score.gender.value(),
        "place_kind_accuracy": score.place_kind.value(),
        "hard_cases_passed": sum(passed for _, passed, _ in score.hard_cases),
        "hard_cases": [
            {"case": case, "passed": passed, "detail": detail}
            for case, passed, detail in score.hard_cases
        ],
        "merge_error_examples": [
            {"entity": e.key, "gold": dict(e.parts)} for e in score.merge_errors[:10]
        ],
        "split_error_examples": [
            {"gold": e.key, "entities": dict(e.parts)} for e in score.split_errors[:10]
        ],
    }


def graph_summary(score: GraphScore) -> dict:
    events = score.events
    return {
        "links": {
            edge_type: tally_summary(tally)
            for edge_type, tally in score.triples.items()
        },
        "siblings_with_shared_parents": tally_summary(score.sibling_closure),
        "places": {
            edge_type: tally_summary(tally) for edge_type, tally in score.places.items()
        },
        "events": {
            "recall": round(events.recall(), 4),
            "matched": events.matched,
            "gold": events.gold_events,
            "by_size": {
                label: {"matched": matched, "gold": total}
                for label, (matched, total) in events.by_size.items()
            },
            "participants": tally_summary(events.participants),
        },
    }


def build_report(
    index_id: str,
    scope: str,
    entities: list[dict],
    edges: list[dict],
    gold: Gold,
    corrections: dict,
    labels_path: Path | None,
) -> dict:
    corrected, corrected_verses = apply_corrections(gold, corrections)
    sections = {}
    for name, key in (("raw", gold), ("corrected", corrected)):
        entity_score = score_entities(
            entities, key.spans, key.tags(scope), key.people_by_slug, key.places_by_slug
        )
        sections[name] = {
            "entities": entity_summary(index_id, entity_score),
            **graph_summary(score_graph(entities, edges, key, scope)),
        }
    narrative = None
    if labels_path is not None and labels_path.exists():
        narrative = score_labels(read_labels(labels_path))
    return {
        "index": index_id,
        "scope": scope,
        "corrected_verses": corrected_verses,
        "dangling_key_links": gold.dangling_links,
        **sections,
        "narrative": narrative,
    }


def render_report(reports: list[dict]) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}"

    def row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |"

    columns = [
        f"{report['index']} {kind}"
        for report in reports
        for kind in ("raw", "corrected")
    ]
    sections = [(report[kind]) for report in reports for kind in ("raw", "corrected")]
    first = reports[0]
    lines = [
        f"# Benchmark against the Theographic answer key ({first['scope']})",
        "",
        "Scores are percentages; F1 combines precision (what we extracted that is right) and recall (what of",
        f"the key we found). Corrected columns apply eval/gold_corrections.yaml ({first['corrected_verses']} verses);",
        f"{first['dangling_key_links']} key links point to ids that do not exist and are ignored throughout.",
        "",
        "## Entities",
        "",
        row("Metric", columns),
        "|---|" + "---|" * len(columns),
        row(
            "B³ F1 (entity resolution)",
            [pct(s["entities"]["bcubed_f1"]) for s in sections],
        ),
        row("Merge errors", [str(s["entities"]["merge_errors"]) for s in sections]),
        row("Split errors", [str(s["entities"]["split_errors"]) for s in sections]),
        row(
            "Person mention F1",
            [pct(s["entities"]["person_mention_f1"]) for s in sections],
        ),
        row(
            "Place mention F1",
            [pct(s["entities"]["place_mention_f1"]) for s in sections],
        ),
        row(
            "Gender accuracy", [pct(s["entities"]["gender_accuracy"]) for s in sections]
        ),
        row(
            "Place kind accuracy",
            [pct(s["entities"]["place_kind_accuracy"]) for s in sections],
        ),
        row(
            "Hard cases passed (of 6)",
            [str(s["entities"]["hard_cases_passed"]) for s in sections],
        ),
        "",
        "## Links (F1, with answer-key count)",
        "",
        row("Link", columns),
        "|---|" + "---|" * len(columns),
    ]
    for edge_type in first["raw"]["links"]:
        lines.append(
            row(
                edge_type,
                [
                    f"{pct(s['links'][edge_type]['f1'])} ({s['links'][edge_type]['gold']})"
                    for s in sections
                ],
            )
        )
    lines.append(
        row(
            "SIBLING_OF incl. shared parents",
            [
                f"{pct(s['siblings_with_shared_parents']['f1'])} ({s['siblings_with_shared_parents']['gold']})"
                for s in sections
            ],
        )
    )
    for edge_type in first["raw"]["places"]:
        lines.append(
            row(
                f"{edge_type} (small sample)",
                [
                    f"{pct(s['places'][edge_type]['f1'])} ({s['places'][edge_type]['gold']})"
                    for s in sections
                ],
            )
        )
    lines += [
        "",
        "## Events",
        "",
        row("Measure", columns),
        "|---|" + "---|" * len(columns),
    ]
    lines.append(
        row(
            "Events found (recall)",
            [
                f"{pct(s['events']['recall'])} ({s['events']['matched']}/{s['events']['gold']})"
                for s in sections
            ],
        )
    )
    for label in first["raw"]["events"]["by_size"]:
        lines.append(
            row(
                f"  {label}",
                [
                    f"{s['events']['by_size'][label]['matched']}/{s['events']['by_size'][label]['gold']}"
                    for s in sections
                ],
            )
        )
    lines.append(
        row(
            "Participant F1 (matched events)",
            [pct(s["events"]["participants"]["f1"]) for s in sections],
        )
    )
    lines += ["", "## Narrative links (hand-labelled)", ""]
    for report in reports:
        narrative = report["narrative"]
        if narrative is None:
            lines.append(
                f"- {report['index']}: not labelled yet (eval/labels/narrative_sample_{report['index']}.csv)"
            )
        else:
            low, high = narrative["interval_95"]
            lines.append(
                f"- {report['index']}: precision {pct(narrative['precision'])} "
                f"(95% interval {pct(low)}–{pct(high)}, {narrative['labeled']} of {narrative['sampled']} labelled; "
                + ", ".join(
                    f"{label} {count}" for label, count in narrative["by_label"].items()
                )
                + ")"
            )
    return "\n".join(lines) + "\n"
