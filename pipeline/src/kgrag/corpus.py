import itertools
import json
import math
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import tiktoken

THEOGRAPHIC_COMMIT = "cfb1c485d4da6fb63a69cb3b7f5b0752792f46bc"
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "theographic"
CORPUS_DIR = DATA_DIR / "corpus"
MAX_PASSAGE_TOKENS = 600

TokenCounter = Callable[[str], int]


def fetch_theographic_file(name: str, raw_dir: Path = RAW_DIR) -> Path:
    path = raw_dir / name
    if path.exists():
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/robertrouse/theographic-bible-metadata/"
        f"{THEOGRAPHIC_COMMIT}/json/{name}"
    )
    partial_path = path.with_suffix(".part")
    with urllib.request.urlopen(url) as response:
        partial_path.write_bytes(response.read())
    partial_path.rename(path)
    return path


def verse_from_raw(record: dict) -> dict:
    # Only the reference and the text may leave this function: the other Theographic
    # fields are the answer key and must never reach extraction.
    fields = record["fields"]
    book, chapter, verse = fields["osisRef"].split(".")
    return {
        "osis_ref": fields["osisRef"],
        "verse_id": fields["verseID"],
        "book": book,
        "chapter": int(chapter),
        "verse": int(verse),
        "text": fields["verseText"].strip(),
    }


def build_verses(raw_records: Iterable[dict]) -> list[dict]:
    return sorted(
        (verse_from_raw(record) for record in raw_records), key=lambda v: v["verse_id"]
    )


def render_verse(verse: dict) -> str:
    return f"[{verse['osis_ref']}] {verse['text']}"


def pack_chapter(
    verses: list[dict], count_tokens: TokenCounter, max_tokens: int
) -> list[list[dict]]:
    # Splitting into equal-sized passages avoids a tiny trailing passage per chapter,
    # which would lose the context the LLM needs to resolve pronouns.
    verse_tokens = [count_tokens(render_verse(verse)) + 1 for verse in verses]
    passage_count = math.ceil(sum(verse_tokens) / max_tokens)
    target_tokens = sum(verse_tokens) / passage_count
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for verse, tokens in zip(verses, verse_tokens, strict=True):
        if current and (
            current_tokens + tokens > max_tokens or current_tokens >= target_tokens
        ):
            groups.append(current)
            current, current_tokens = [], 0
        current.append(verse)
        current_tokens += tokens
    groups.append(current)
    return groups


def pack_passages(
    verses: list[dict], count_tokens: TokenCounter, max_tokens: int = MAX_PASSAGE_TOKENS
) -> list[dict]:
    passages = []
    for (book, chapter), chapter_verses in itertools.groupby(
        verses, key=lambda v: (v["book"], v["chapter"])
    ):
        for index, group in enumerate(
            pack_chapter(list(chapter_verses), count_tokens, max_tokens), start=1
        ):
            text = "\n".join(render_verse(verse) for verse in group)
            passages.append(
                {
                    "id": f"{book}.{chapter}.p{index}",
                    "book": book,
                    "chapter": chapter,
                    "verse_refs": [verse["osis_ref"] for verse in group],
                    "text": text,
                    "token_count": count_tokens(text),
                }
            )
    return passages


def tiktoken_counter() -> TokenCounter:
    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode(text))


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)
