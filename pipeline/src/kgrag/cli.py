import json
import os
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from kgrag.build import (
    INDEX_DIR,
    add_graph_layout,
    add_overview_ranks,
    build_index,
)
from kgrag.confirm import PURPOSE as CONFIRM_PURPOSE
from kgrag.confirm import llm_confirmer
from kgrag.corpus import (
    CORPUS_DIR,
    MAX_PASSAGE_TOKENS,
    REPO_ROOT,
    build_verses,
    fetch_theographic_file,
    pack_passages,
    read_jsonl,
    read_passage_ids,
    tiktoken_counter,
    write_jsonl,
)
from kgrag.embed import (
    embed_texts,
    entity_profile,
    load_embeddings,
    passage_body,
    save_embeddings,
)
from kgrag.extract import (
    PURPOSE,
    extract_passages,
    extraction_dir,
    load_extractions,
    load_system_prompt,
    user_prompt,
)
from kgrag.infer import PURPOSE as PARENT_PURPOSE
from kgrag.infer import infer_parent_links, llm_parent_confirmer
from kgrag.kinship import PURPOSE as KINSHIP_PURPOSE
from kgrag.kinship import llm_kinship_checker, relabel_loose_kinship
from kgrag.llm import (
    Budget,
    BudgetExceeded,
    ClaudeCodeClient,
    Ledger,
    ModelConfig,
    summarize_usage,
)
from kgrag.neo4j_load import connect, graph_counts, load_graph
from kgrag.resolve import AliasRules, resolve_naive, resolve_v1
from kgrag.schema import PassageExtraction

DEFAULT_PROJECT_BUDGET_USD = 40.0
CONFIRM_ABOVE_USD = 1.0

app = typer.Typer(no_args_is_help=True)
corpus_app = typer.Typer(
    no_args_is_help=True, help="Fetch the KJV text and split it into passages."
)
app.add_typer(corpus_app, name="corpus")


@app.callback()
def main() -> None:
    """Bible Graph RAG batch pipeline."""
    load_dotenv(REPO_ROOT / ".env")


def load_alias_rules() -> AliasRules:
    rules = (
        yaml.safe_load(
            (REPO_ROOT / "config" / "aliases.yaml").read_text(encoding="utf-8")
        )
        or {}
    )
    return AliasRules(
        must_link=tuple(tuple(pair) for pair in rules.get("must_link") or []),
        cannot_link=tuple(tuple(pair) for pair in rules.get("cannot_link") or []),
    )


def project_budget_usd() -> float:
    return float(os.environ.get("KGRAG_PROJECT_BUDGET_USD", DEFAULT_PROJECT_BUDGET_USD))


@corpus_app.command()
def fetch() -> None:
    path = fetch_theographic_file("verses.json")
    typer.echo(f"{path} ({path.stat().st_size:,} bytes)")


@corpus_app.command()
def build() -> None:
    raw_records = json.loads(
        fetch_theographic_file("verses.json").read_text(encoding="utf-8")
    )
    verses = build_verses(raw_records)
    passages = pack_passages(verses, tiktoken_counter())
    write_jsonl(CORPUS_DIR / "verses.jsonl", verses)
    write_jsonl(CORPUS_DIR / "passages.jsonl", passages)

    chapters = {(verse["book"], verse["chapter"]) for verse in verses}
    genesis_verses = sum(1 for verse in verses if verse["book"] == "Gen")
    genesis_passages = sum(1 for passage in passages if passage["book"] == "Gen")
    largest_passage = max(passage["token_count"] for passage in passages)
    typer.echo(f"verses:   {len(verses):,} (Genesis {genesis_verses:,})")
    typer.echo(f"chapters: {len(chapters):,}")
    typer.echo(f"passages: {len(passages):,} (Genesis {genesis_passages:,})")
    typer.echo(
        f"largest passage: {largest_passage} tokens (limit {MAX_PASSAGE_TOKENS})"
    )


@corpus_app.command()
def show(passage_id: str) -> None:
    for passage in read_jsonl(CORPUS_DIR / "passages.jsonl"):
        if passage["id"] == passage_id:
            typer.echo(passage["text"])
            return
    raise typer.BadParameter(f"no passage with id {passage_id!r}")


def select_passages(passages_file: Path | None, scope: str | None) -> list[dict]:
    passages = list(read_jsonl(CORPUS_DIR / "passages.jsonl"))
    if passages_file is not None:
        wanted = read_passage_ids(passages_file)
        by_id = {passage["id"]: passage for passage in passages}
        missing = [passage_id for passage_id in wanted if passage_id not in by_id]
        if missing:
            raise typer.BadParameter(f"unknown passage ids: {', '.join(missing)}")
        return [by_id[passage_id] for passage_id in wanted]
    if scope is not None:
        return [passage for passage in passages if scope in ("Bible", passage["book"])]
    raise typer.BadParameter("pass --passages FILE or --scope BOOK")


@app.command()
def extract(
    passages_file: Path | None = typer.Option(
        None, "--passages", help="File with one passage id per line."
    ),
    scope: str | None = typer.Option(
        None, help="OSIS book code such as Gen, or Bible."
    ),
    model: str = typer.Option(
        "claude-haiku-5-5", help="Full model id; aliases drift over time."
    ),
    effort: str = typer.Option("medium", help="low, medium, high, xhigh or max."),
    run_budget_usd: float = typer.Option(
        2.0, help="Stop this run once it has spent this much."
    ),
    call_budget_usd: float = typer.Option(
        0.10, help="Cap for a single claude -p call."
    ),
    workers: int = typer.Option(4, help="Parallel claude -p calls."),
    dry_run: bool = typer.Option(
        False, help="Show what would run and the projected cost, then stop."
    ),
    yes: bool = typer.Option(
        False, "--yes", help=f"Skip the confirmation above ${CONFIRM_ABOVE_USD:.2f}."
    ),
) -> None:
    passages = select_passages(passages_file, scope)
    config = ModelConfig(model=model, effort=effort)
    ledger = Ledger()
    system_prompt = load_system_prompt()
    probe = ClaudeCodeClient(
        config, budget=None, ledger=ledger, run_id="", purpose=PURPOSE
    )
    to_call = [
        passage
        for passage in passages
        if not probe.is_cached(system_prompt, user_prompt(passage), PassageExtraction)
    ]
    average_cost = ledger.average_call_cost(config, PURPOSE)
    projected = None if average_cost is None else average_cost * len(to_call)
    spent_total = ledger.total_cost()

    typer.echo(f"model:      {model} (effort {effort})")
    typer.echo(
        f"passages:   {len(passages)} selected, {len(passages) - len(to_call)} cached, {len(to_call)} to call"
    )
    typer.echo(
        "projected:  "
        + (
            f"${projected:.4f}"
            if projected is not None
            else "unknown, no history for this model/effort yet"
        )
    )
    typer.echo(f"run cap:    ${run_budget_usd:.2f} (per call ${call_budget_usd:.2f})")
    typer.echo(f"project:    ${spent_total:.4f} spent of ${project_budget_usd():.2f}")
    if dry_run:
        return

    worst_case = projected if projected is not None else run_budget_usd
    if to_call and worst_case > CONFIRM_ABOVE_USD and not yes:
        typer.confirm(
            f"This run may spend up to ${min(worst_case, run_budget_usd):.2f}. Continue?",
            abort=True,
        )

    try:
        budget = (
            Budget(run_budget_usd, project_budget_usd(), call_budget_usd, spent_total)
            if to_call
            else None
        )
    except BudgetExceeded as error:
        raise typer.Exit(f"stopped: {error}") from error
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    client = ClaudeCodeClient(
        config, budget=budget, ledger=ledger, run_id=run_id, purpose=PURPOSE
    )
    summary = extract_passages(client, passages, extraction_dir(config), workers)

    typer.echo(f"extracted:  {summary.extracted} new, {summary.from_cache} from cache")
    if summary.failed:
        typer.echo(f"failed:     {len(summary.failed)}")
        for failure in summary.failed:
            typer.echo(f"  - {failure}")
    if summary.stopped_by_budget:
        typer.echo(
            f"stopped:    {summary.stopped_by_budget}; {summary.not_attempted} passages left, rerun to continue"
        )
    spent_run = budget.spent_this_run_usd if budget else 0.0
    typer.echo(
        f"spent:      ${spent_run:.4f} this run, ${spent_total + spent_run:.4f} of ${project_budget_usd():.2f} in total"
    )
    typer.echo(f"output:     {extraction_dir(config)}")


@app.command()
def usage() -> None:
    """Show LLM spending from the ledger, in API-equivalent USD."""
    rows = Ledger().rows()
    now = datetime.now(UTC)
    for title, since in [
        ("today", now.replace(hour=0, minute=0, second=0)),
        ("last 7 days", now - timedelta(days=7)),
        ("all time", None),
    ]:
        totals = summarize_usage(rows, since)
        typer.echo(f"\n{title}")
        if not totals:
            typer.echo("  no calls")
            continue
        typer.echo(
            f"  {'model':<26} {'effort':<7} {'purpose':<9} {'calls':>6} {'in tok':>10} {'out tok':>9} {'think':>8} {'USD':>9}"
        )
        for (model, effort, purpose), group in sorted(totals.items()):
            typer.echo(
                f"  {model:<26} {effort:<7} {purpose:<9} {group['calls']:>6,} {group['input_tokens']:>10,} "
                f"{group['output_tokens']:>9,} {group['thinking_tokens']:>8,} {group['cost_usd']:>9.4f}"
            )
    spent = sum(row["cost_usd"] for row in rows)
    typer.echo(
        f"\nproject budget: ${spent:.4f} spent of ${project_budget_usd():.2f} (${project_budget_usd() - spent:.4f} left)"
    )


@app.command("build")
def build_graph(
    scope: str = typer.Option("Gen", help="OSIS book code such as Gen, or Bible."),
    model: str = typer.Option("claude-haiku-5-5"),
    effort: str = typer.Option("medium"),
    resolver: str = typer.Option(
        "v1", help="naive (exact names) or v1 (rules plus LLM confirmation)."
    ),
    run_budget_usd: float = typer.Option(
        1.0, help="Cap for the v1 LLM confirmation calls."
    ),
) -> None:
    """Merge passage extractions into one graph, with layout and clusters."""
    passage_ids = [
        passage["id"]
        for passage in read_jsonl(CORPUS_DIR / "passages.jsonl")
        if scope in ("Bible", passage["book"])
    ]
    config = ModelConfig(model=model, effort=effort)
    extractions, missing = load_extractions(config, passage_ids)
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} passages not extracted yet, e.g. {missing[0]}; run kgrag extract first"
        )
    verse_order = {
        verse["osis_ref"]: index
        for index, verse in enumerate(read_jsonl(CORPUS_DIR / "verses.jsonl"))
    }
    index_id = f"{scope.lower()}-{resolver}"
    out_dir = INDEX_DIR / index_id
    if resolver == "naive":
        resolution = resolve_naive(extractions)
    elif resolver == "v1":
        ledger = Ledger()
        budget = Budget(run_budget_usd, project_budget_usd(), 0.10, ledger.total_cost())
        client = ClaudeCodeClient(
            ModelConfig(model=model, effort=effort),
            budget=budget,
            ledger=ledger,
            run_id=f"er-{datetime.now(UTC):%Y%m%dT%H%M%S}",
            purpose=CONFIRM_PURPOSE,
        )
        verse_text = {
            verse["osis_ref"]: verse["text"]
            for verse in read_jsonl(CORPUS_DIR / "verses.jsonl")
        }
        resolution, decisions = resolve_v1(
            extractions,
            verse_order,
            llm_confirmer(client, verse_text),
            load_alias_rules(),
        )
        write_jsonl(
            out_dir / "er_audit.jsonl", (asdict(decision) for decision in decisions)
        )
        rules = Counter((decision.verdict, decision.rule) for decision in decisions)
        typer.echo("resolution decisions:")
        for (verdict, rule), count in rules.most_common():
            typer.echo(f"  {verdict:<7} {count:>5}  {rule}")
        typer.echo(f"confirmation spend: ${budget.spent_this_run_usd:.4f}")
    else:
        raise typer.BadParameter(f"unknown resolver {resolver!r}; use naive or v1")
    entities, edges = build_index(extractions, resolution, verse_order, layout=False)
    if resolver == "v1":
        names = {entity["id"]: entity["name"] for entity in entities}
        kinship_client = ClaudeCodeClient(
            ModelConfig(model=model, effort=effort),
            budget=budget,
            ledger=ledger,
            run_id=client.run_id,
            purpose=KINSHIP_PURPOSE,
        )
        edges, kinship_decisions = relabel_loose_kinship(
            edges, llm_kinship_checker(kinship_client, names, verse_text)
        )
        write_jsonl(out_dir / "kinship_audit.jsonl", kinship_decisions)
        loose = sum(decision["verdict"] == "loose" for decision in kinship_decisions)
        typer.echo(
            f"kinship words read loosely: {loose} of {len(kinship_decisions)} flagged"
        )
        parent_client = ClaudeCodeClient(
            ModelConfig(model=model, effort=effort),
            budget=budget,
            ledger=ledger,
            run_id=client.run_id,
            purpose=PARENT_PURPOSE,
        )
        added, parent_decisions = infer_parent_links(
            entities,
            edges,
            verse_order,
            llm_parent_confirmer(parent_client, names, verse_text),
        )
        edges = sorted(edges + added, key=lambda edge: edge["id"])
        write_jsonl(out_dir / "parent_audit.jsonl", parent_decisions)
        typer.echo(
            f"parent links inferred: {len(added)} of {len(parent_decisions)} proposed"
        )
        typer.echo(f"confirmation spend, both steps: ${budget.spent_this_run_usd:.4f}")
    add_graph_layout(entities, edges)
    add_overview_ranks(entities, edges, verse_order)

    write_jsonl(out_dir / "entities.jsonl", entities)
    write_jsonl(out_dir / "edges.jsonl", edges)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "index_id": index_id,
                "scope": scope,
                "resolver": resolver,
                "extraction_model": model,
                "extraction_effort": effort,
                "passages": len(extractions),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    kinds = Counter(entity["kind"] for entity in entities)
    types = Counter(edge["type"] for edge in edges)
    typer.echo(
        f"entities:    {len(entities):,} ({', '.join(f'{kind} {count}' for kind, count in kinds.most_common())})"
    )
    typer.echo(f"edges:       {len(edges):,}")
    for edge_type, count in types.most_common():
        typer.echo(f"  {edge_type:<16} {count:>5}")
    typer.echo(f"communities: {len({entity['community'] for entity in entities})}")
    typer.echo(f"output:      {out_dir}")


def index_scope(index_id: str) -> tuple[dict, list[dict]]:
    meta = json.loads((INDEX_DIR / index_id / "meta.json").read_text(encoding="utf-8"))
    passages = [
        passage
        for passage in read_jsonl(CORPUS_DIR / "passages.jsonl")
        if meta["scope"] in ("Bible", passage["book"])
    ]
    return meta, passages


@app.command()
def embed(index_id: str = typer.Option("gen-naive", "--index")) -> None:
    """Embed entity profiles and passages with the model the browser will also run."""
    _, passages = index_scope(index_id)
    entities = list(read_jsonl(INDEX_DIR / index_id / "entities.jsonl"))
    save_embeddings(
        INDEX_DIR / index_id / "entity_embeddings.npz",
        [entity["id"] for entity in entities],
        embed_texts([entity_profile(entity) for entity in entities]),
    )
    save_embeddings(
        INDEX_DIR / index_id / "passage_embeddings.npz",
        [passage["id"] for passage in passages],
        embed_texts([passage_body(passage["text"]) for passage in passages]),
    )
    typer.echo(f"embedded {len(entities):,} entities and {len(passages):,} passages")


@app.command("neo4j-load")
def neo4j_load(index_id: str = typer.Option("gen-naive", "--index")) -> None:
    """Wipe the local Neo4j database and load one index into it."""
    meta, passages = index_scope(index_id)
    verse_refs = {
        verse_ref for passage in passages for verse_ref in passage["verse_refs"]
    }
    verses = [
        verse
        for verse in read_jsonl(CORPUS_DIR / "verses.jsonl")
        if verse["osis_ref"] in verse_refs
    ]
    with connect() as driver:
        load_graph(
            driver,
            meta,
            verses,
            passages,
            list(read_jsonl(INDEX_DIR / index_id / "entities.jsonl")),
            list(read_jsonl(INDEX_DIR / index_id / "edges.jsonl")),
            load_embeddings(INDEX_DIR / index_id / "entity_embeddings.npz"),
            load_embeddings(INDEX_DIR / index_id / "passage_embeddings.npz"),
        )
        for name, count in graph_counts(driver).items():
            typer.echo(f"{name:<22} {count:>8,}")
