import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml

from kgrag.build import INDEX_DIR
from kgrag.cli import project_budget_usd
from kgrag.corpus import CORPUS_DIR, REPO_ROOT, read_jsonl, read_passage_ids
from kgrag.extract import extraction_dir, load_extractions
from kgrag.llm import Budget, BudgetExceeded, ClaudeCodeClient, Ledger, ModelConfig
from kgrag_eval.bakeoff import ConfigScore, load_extraction, score_passage
from kgrag_eval.entities import align_mentions, majority, score_entities
from kgrag_eval.fixture import FIXTURE_DIR, make_fixture
from kgrag_eval.fixture import SCOPE as FIXTURE_SCOPE
from kgrag_eval.narrative import sample_narrative_edges, write_sample
from kgrag_eval.report import build_report, entity_summary, render_report
from kgrag_eval.score_qa import (
    JUDGE_PURPOSE,
    check_grounding,
    judge_rows,
    lineage_gold,
    load_run,
    read_human_grades,
    score_lineage,
    score_trace,
    summarize,
    write_grades,
)
from kgrag_eval.theographic import (
    ALL_BOOKS,
    apply_corrections,
    compute_gold_stats,
    load_gold,
    load_records,
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
    gold = load_gold()

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
                    gold.spans,
                    gold.people_by_slug,
                    gold.places_by_slug,
                    gold.person_slug_by_id,
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

    markdown = render_markdown(
        rows, recommendation, len(passage_ids), gold.dangling_links
    )
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


def render_entity_markdown(
    summaries: list[dict], scope: str, dangling_links: int
) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}"

    metrics = [
        ("B³ precision", "bcubed_precision", pct),
        ("B³ recall", "bcubed_recall", pct),
        ("B³ F1", "bcubed_f1", pct),
        ("Merge errors (one entity, several real people)", "merge_errors", str),
        ("Split errors (one real person, several entities)", "split_errors", str),
        ("Person mention F1", "person_mention_f1", pct),
        ("Place mention F1", "place_mention_f1", pct),
        ("Gender accuracy", "gender_accuracy", pct),
        ("Place kind accuracy", "place_kind_accuracy", pct),
        ("Hard cases passed (of 6)", "hard_cases_passed", str),
        ("Aligned mentions", "aligned_mentions", str),
    ]
    lines = [
        f"# Entity resolution scoreboard ({scope})",
        "",
        "B³ scores each name mention by how cleanly its entity matches the real person or place the",
        "answer key says it is. Precision falls when people are merged; recall falls when one person is",
        f"split. {dangling_links} answer-key links point to ids that do not exist and are ignored.",
        "",
        "| Metric | " + " | ".join(summary["index"] for summary in summaries) + " |",
        "|---|" + "---|" * len(summaries),
    ]
    for label, key, fmt in metrics:
        lines.append(
            f"| {label} | "
            + " | ".join(fmt(summary[key]) for summary in summaries)
            + " |"
        )
    for summary in summaries:
        lines += [
            "",
            f"## {summary['index']}: hard cases",
            "",
            "| Case | Passed | Entities |",
            "|---|---|---|",
        ]
        lines += [
            f"| {c['case']} | {'yes' if c['passed'] else 'no'} | {c['detail']} |"
            for c in summary["hard_cases"]
        ]
        lines += [
            "",
            "Largest merge errors: "
            + "; ".join(
                f"{e['entity']} = "
                + ", ".join(f"{g} {n}" for g, n in e["gold"].items())
                for e in summary["merge_error_examples"][:5]
            ),
        ]
        lines += [
            "",
            "Largest split errors: "
            + "; ".join(
                f"{e['gold']} = "
                + ", ".join(f"{g} {n}" for g, n in e["entities"].items())
                for e in summary["split_error_examples"][:5]
            ),
        ]
    return "\n".join(lines) + "\n"


@app.command()
def er(
    index_ids: list[str] = typer.Option(
        ..., "--index", help="Index id such as gen-naive; repeat to compare."
    ),
) -> None:
    """Score entity resolution of one or more graph indexes against the answer key."""
    gold = load_gold()
    summaries = []
    scopes = set()
    for index_id in index_ids:
        meta = json.loads(
            (INDEX_DIR / index_id / "meta.json").read_text(encoding="utf-8")
        )
        scopes.add(meta["scope"])
        entities = list(read_jsonl(INDEX_DIR / index_id / "entities.jsonl"))
        score = score_entities(
            entities,
            gold.spans,
            gold.tags(meta["scope"]),
            gold.people_by_slug,
            gold.places_by_slug,
        )
        summaries.append(entity_summary(index_id, score))
    scope = ", ".join(sorted(scopes))
    markdown = render_entity_markdown(summaries, scope, gold.dangling_links)
    out_dir = (
        REPO_ROOT / "eval" / "results" / "er" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(markdown, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(summaries, indent=1), encoding="utf-8"
    )
    typer.echo(markdown)
    typer.echo(f"written to {out_dir}")


def load_index(index_id: str) -> tuple[dict, list[dict], list[dict]]:
    meta = json.loads((INDEX_DIR / index_id / "meta.json").read_text(encoding="utf-8"))
    return (
        meta,
        list(read_jsonl(INDEX_DIR / index_id / "entities.jsonl")),
        list(read_jsonl(INDEX_DIR / index_id / "edges.jsonl")),
    )


@app.command()
def report(
    index_ids: list[str] = typer.Option(
        ..., "--index", help="Index id such as gen-v1; repeat to compare."
    ),
) -> None:
    """Full benchmark: entities, links, places, events and labelled narrative links, raw and corrected."""
    gold = load_gold()
    corrections = (
        yaml.safe_load(
            (REPO_ROOT / "eval" / "gold_corrections.yaml").read_text(encoding="utf-8")
        )
        or {}
    )
    reports = []
    for index_id in index_ids:
        meta, entities, edges = load_index(index_id)
        reports.append(
            build_report(
                index_id,
                meta["scope"],
                entities,
                edges,
                gold,
                corrections,
                REPO_ROOT / "eval" / "labels" / f"narrative_sample_{index_id}.csv",
            )
        )
    markdown = render_report(reports)
    out_dir = (
        REPO_ROOT / "eval" / "results" / "report" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(markdown, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(reports, indent=1), encoding="utf-8"
    )
    (REPO_ROOT / "eval" / "results" / "latest.json").write_text(
        json.dumps(reports, indent=1), encoding="utf-8"
    )
    typer.echo(markdown)
    typer.echo(f"written to {out_dir} and eval/results/latest.json")


@app.command("narrative-sample")
def narrative_sample(
    index_id: str = typer.Option("gen-v1", "--index"),
    size: int = typer.Option(100, help="How many narrative links to sample."),
    seed: int = typer.Option(7, help="Fixed so the same sample comes back."),
    force: bool = typer.Option(
        False, help="Overwrite an existing sample, labels included."
    ),
) -> None:
    """Write a CSV of random narrative links for hand-labelling."""
    path = REPO_ROOT / "eval" / "labels" / f"narrative_sample_{index_id}.csv"
    if path.exists() and not force:
        raise typer.BadParameter(
            f"{path} exists and may hold labels; pass --force to replace it"
        )
    _, entities, edges = load_index(index_id)
    verse_text = {
        verse["osis_ref"]: verse["text"]
        for verse in read_jsonl(CORPUS_DIR / "verses.jsonl")
    }
    rows = sample_narrative_edges(entities, edges, verse_text, size, seed)
    write_sample(path, rows)
    typer.echo(f"wrote {len(rows)} links to {path}")
    typer.echo(
        "label each row in the label column: correct, wrong_type, wrong_direction or unsupported"
    )


@app.command()
def fixture(
    index_id: str = typer.Option("gen-v1", "--index"),
    model: str = typer.Option("claude-haiku-5-5"),
    effort: str = typer.Option("medium"),
) -> None:
    """Regenerate the Genesis regression fixture and its expected report after an intended change."""
    passage_ids = [
        p["id"]
        for p in read_jsonl(CORPUS_DIR / "passages.jsonl")
        if p["book"] == FIXTURE_SCOPE
    ]
    extractions, missing = load_extractions(ModelConfig(model, effort), passage_ids)
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} passages not extracted, e.g. {missing[0]}"
        )
    records = {
        name: load_records(f"{name}.json")
        for name in ("verses", "people", "places", "events")
    }
    make_fixture(
        extractions,
        INDEX_DIR / index_id / "er_audit.jsonl",
        INDEX_DIR / index_id / "parent_audit.jsonl",
        records,
    )
    typer.echo(f"fixture written to {FIXTURE_DIR}")


@app.command()
def qa(
    run_id: str = typer.Option(..., "--run", help="Folder name under eval/runs."),
    judge_model: str = typer.Option("claude-sonnet-5-5"),
    judge_effort: str = typer.Option("medium"),
    index_id: str = typer.Option("gen-v1", "--index"),
    run_budget_usd: float = typer.Option(1.0),
    workers: int = typer.Option(4),
) -> None:
    """Score a run of eval/questions.yaml: LLM-judged accuracy, family trees shown, sentences backed by sources, verse recall, tokens, latency and cost."""
    questions = {
        question["id"]: question
        for question in yaml.safe_load(
            (REPO_ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8")
        )
    }
    passages = list(read_jsonl(CORPUS_DIR / "passages.jsonl"))
    passage_verses = {passage["id"]: passage["verse_refs"] for passage in passages}
    passage_text = {passage["id"]: passage["text"] for passage in passages}
    verse_text = {
        verse["osis_ref"]: verse["text"]
        for verse in read_jsonl(CORPUS_DIR / "verses.jsonl")
    }
    corrections = (
        yaml.safe_load(
            (REPO_ROOT / "eval" / "gold_corrections.yaml").read_text(encoding="utf-8")
        )
        or {}
    )
    gold, _ = apply_corrections(load_gold(), corrections)
    _, entities, edges = load_index(index_id)
    names = {entity["id"]: entity["name"] for entity in entities}
    # What the model was shown for each entity and link it could cite.
    card_text = {
        entity["id"]: f"{entity['name']} ({entity['kind']}): {entity['description']}"
        for entity in entities
    } | {
        edge[
            "id"
        ]: f"{names[edge['source']]} {edge['type']} {names[edge['target']]}: {edge['description']}"
        for edge in edges
    }
    gold_of_entity = majority(
        align_mentions(entities, "person", gold.spans), "entity", "gold"
    )
    names_of: dict[str, set[str]] = defaultdict(set)
    our_name_of = {}
    for slug, person in gold.people_by_slug.items():
        names_of[slug].add(person["name"])
    for entity in entities:
        if entity["id"] in gold_of_entity:
            slug = gold_of_entity[entity["id"]]
            our_name_of.setdefault(slug, entity["name"])
            names_of[slug].update(
                name
                for name in [entity["name"], *entity["aliases"]]
                if name[:1].isupper()
            )

    def display(slug: str) -> str:
        name = gold.people_by_slug[slug]["name"]
        ours = our_name_of.get(slug, name)
        return name if ours == name else f"{name} ({ours})"

    lineages = {}
    for question in questions.values():
        if "lineage" in question:
            root = question["lineage"]["person"]
            direction = question["lineage"]["direction"]
            people, links = lineage_gold(gold, root, direction, FIXTURE_SCOPE)
            lineages[question["id"]] = (root, people, links)
            question["answer"] = (
                f"The {direction} of {display(root)} named in Genesis ({len(people)} people): "
                + ", ".join(sorted(display(slug) for slug in people))
                + "."
            )
    pairs = []
    for trace in load_run(REPO_ROOT / "eval" / "runs" / run_id):
        question = questions[trace_question_id(trace, questions)]
        row = score_trace(trace, question, passage_verses)
        if question["id"] in lineages:
            root, people, links = lineages[question["id"]]
            row |= score_lineage(
                trace, row["answer"], root, people, links, gold_of_entity, names_of
            )
        pairs.append((row, trace))
    rows = [row for row, _ in pairs]
    ledger = Ledger()
    try:
        budget = Budget(run_budget_usd, project_budget_usd(), 0.10, ledger.total_cost())
    except BudgetExceeded as error:
        raise typer.Exit(f"stopped: {error}") from error
    client = ClaudeCodeClient(
        ModelConfig(judge_model, judge_effort),
        budget=budget,
        ledger=ledger,
        run_id=f"qa-{run_id}",
        purpose=JUDGE_PURPOSE,
    )
    judge_rows(client, rows, questions, workers)
    check_grounding(client, pairs, verse_text, passage_text, card_text, workers)
    grades_path = REPO_ROOT / "eval" / "labels" / f"qa_grades_{run_id}.csv"
    human = read_human_grades(grades_path)
    write_grades(grades_path, rows, questions, human)
    summary = {
        "run": run_id,
        "judge": f"{judge_model} {judge_effort}",
        "judge_cost_usd": round(
            sum(row["judge_cost_usd"] + row["grounding_cost_usd"] for row in rows), 4
        ),
        **summarize(rows, human),
    }
    out_dir = REPO_ROOT / "eval" / "results" / "qa" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (REPO_ROOT / "eval" / "results" / "qa_latest.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    typer.echo(json.dumps(summary, indent=1))
    typer.echo(f"grades for review: {grades_path.relative_to(REPO_ROOT)}")


def trace_question_id(trace: dict, questions: dict[str, dict]) -> str:
    return next(
        question_id
        for question_id, question in questions.items()
        if question["question"] == trace["query"]["text"]
    )
