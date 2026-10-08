from dataclasses import asdict

import typer

from kgrag_eval.theographic import ALL_BOOKS, compute_gold_stats, load_records

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
