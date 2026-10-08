import json
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from kgrag.corpus import CORPUS_DIR, REPO_ROOT, read_jsonl, read_passage_ids
from kgrag.extract import extraction_dir
from kgrag.llm import ModelConfig
from kgrag_eval.bakeoff import ConfigScore, load_extraction, score_passage
from kgrag_eval.theographic import (
    ALL_BOOKS,
    compute_gold_stats,
    load_records,
    parse_md_mentions,
)

COMBINED_TOLERANCE = 0.03

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Grade the extracted graph against the Theographic answer key."""


@app.command("gold-stats")
def gold_stats(
    scope: str = typer.Option(ALL_BOOKS, help="OSIS book code such as Gen, or Bible."),
) -> None:
    stats = compute_gold_stats(
        load_records("verses.json"), load_records("people.json"), scope
    )
    for field, value in asdict(stats).items():
        typer.echo(f"{field.replace('_', ' '):<20} {value:>8,}")


def score_row(name: str, score: ConfigScore) -> dict:
    return {
        "config": name,
        "passages": score.passages,
        "person_f1": score.person_mentions.f1(),
        "person_precision": score.person_mentions.precision(),
        "person_recall": score.person_mentions.recall(),
        "place_f1": score.place_mentions.f1(),
        "place_precision": score.place_mentions.precision(),
        "place_recall": score.place_mentions.recall(),
        "parent_f1": score.parent_links.f1(),
        "parent_precision": score.parent_links.precision(),
        "parent_recall": score.parent_links.recall(),
        "spouse_f1": score.spouse_links.f1(),
        "sibling_f1": score.sibling_links.f1(),
        "gender_accuracy": score.gender.value(),
        "place_kind_accuracy": score.place_kind.value(),
        "combined": score.combined(),
        "drop_rate": score.dropped_items / score.raw_items if score.raw_items else 0.0,
        "cost_per_passage_usd": statistics.fmean(score.costs_usd)
        if score.costs_usd
        else 0.0,
        "total_cost_usd": sum(score.costs_usd),
        "median_seconds": statistics.median(score.durations_ms) / 1000
        if score.durations_ms
        else 0.0,
        "thinking_share": score.thinking_tokens / score.output_tokens
        if score.output_tokens
        else 0.0,
    }


def render_markdown(
    rows: list[dict], recommendation: str, passage_count: int, dangling_links: int
) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}"

    lines = [
        f"# Extraction bake-off ({passage_count} Genesis passages)",
        "",
        "Scores are percentages. Mentions are (verse, entity) pairs after mapping each extracted name to",
        "the answer key within its passage. Family recall is a lower bound: the answer key also lists",
        "relations that a passage names both people for but never states. "
        f"{dangling_links} answer-key links point to ids that do not exist and are ignored.",
        "",
        "| config | n | person F1 (P/R) | place F1 (P/R) | parent F1 (P/R) | spouse F1 | sibling F1 "
        "| gender | place kind | combined | dropped | $/passage | median s | thinking |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['passages']} "
            f"| {pct(row['person_f1'])} ({pct(row['person_precision'])}/{pct(row['person_recall'])}) "
            f"| {pct(row['place_f1'])} ({pct(row['place_precision'])}/{pct(row['place_recall'])}) "
            f"| {pct(row['parent_f1'])} ({pct(row['parent_precision'])}/{pct(row['parent_recall'])}) "
            f"| {pct(row['spouse_f1'])} | {pct(row['sibling_f1'])} "
            f"| {pct(row['gender_accuracy'])} | {pct(row['place_kind_accuracy'])} | {pct(row['combined'])} "
            f"| {pct(row['drop_rate'])} | {row['cost_per_passage_usd']:.4f} | {row['median_seconds']:.1f} "
            f"| {pct(row['thinking_share'])} |"
        )
    lines += ["", f"**Recommendation:** {recommendation}", ""]
    return "\n".join(lines)


@app.command()
def bakeoff(
    configs: list[str] = typer.Option(
        ..., "--config", help="model:effort; repeat for each config."
    ),
    passages_file: Path = typer.Option(
        REPO_ROOT / "eval" / "bakeoff_passages.txt", "--passages"
    ),
) -> None:
    passage_ids = read_passage_ids(passages_file)
    passages = {
        passage["id"]: passage for passage in read_jsonl(CORPUS_DIR / "passages.jsonl")
    }
    people = load_records("people.json")
    places = load_records("places.json")
    people_by_slug = {fields["slug"]: fields for fields in people.values()}
    places_by_slug = {fields["slug"]: fields for fields in places.values()}
    slug_by_id = {record_id: fields["slug"] for record_id, fields in people.items()}
    known_slugs = {"person": people_by_slug, "place": places_by_slug}
    gold_spans = {}
    dangling_links = 0
    for fields in load_records("verses.json").values():
        spans = parse_md_mentions(fields.get("mdText", ""))
        gold_spans[fields["osisRef"]] = [
            span for span in spans if span.slug in known_slugs[span.kind]
        ]
        dangling_links += len(spans) - len(gold_spans[fields["osisRef"]])

    rows = []
    for config_text in configs:
        model, effort = config_text.rsplit(":", 1)
        score = ConfigScore()
        for passage_id in passage_ids:
            extraction = load_extraction(
                extraction_dir(ModelConfig(model, effort)), passage_id
            )
            if extraction is not None:
                score_passage(
                    score,
                    extraction,
                    passages[passage_id]["text"],
                    gold_spans,
                    people_by_slug,
                    places_by_slug,
                    slug_by_id,
                )
        rows.append(score_row(config_text, score))

    complete = [row for row in rows if row["passages"] == len(passage_ids)]
    if complete:
        best = max(row["combined"] for row in complete)
        cheapest = min(
            (row for row in complete if row["combined"] >= best - COMBINED_TOLERANCE),
            key=lambda row: row["cost_per_passage_usd"],
        )
        recommendation = (
            f"{cheapest['config']}: the cheapest config within {COMBINED_TOLERANCE * 100:.0f} points "
            f"of the best combined score ({best * 100:.1f})."
        )
    else:
        recommendation = "no config has extractions for every passage yet."

    markdown = render_markdown(rows, recommendation, len(passage_ids), dangling_links)
    out_dir = (
        REPO_ROOT
        / "eval"
        / "results"
        / "bakeoff"
        / f"{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(markdown, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    typer.echo(markdown)
    typer.echo(f"written to {out_dir}")
