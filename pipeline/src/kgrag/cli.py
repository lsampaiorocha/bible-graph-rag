import json

import typer

from kgrag.corpus import (
    CORPUS_DIR,
    MAX_PASSAGE_TOKENS,
    build_verses,
    fetch_theographic_file,
    pack_passages,
    read_jsonl,
    tiktoken_counter,
    write_jsonl,
)

app = typer.Typer(no_args_is_help=True)
corpus_app = typer.Typer(
    no_args_is_help=True, help="Fetch the KJV text and split it into passages."
)
app.add_typer(corpus_app, name="corpus")


@app.callback()
def main() -> None:
    """Bible Graph RAG batch pipeline."""


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
