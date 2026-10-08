import csv
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from kgrag.llm import ClaudeCodeClient

JUDGE_PURPOSE = "qa_judge"
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "no_answer": 0.0}
GRADE_FIELDS = [
    "qid",
    "category",
    "mode",
    "question",
    "gold",
    "answer",
    "judge",
    "judge_reason",
    "human",
]

JUDGE_SYSTEM = """You grade answers to questions about the book of Genesis (King James Version) against a reference answer.

Verdicts:
- correct: states the reference answer's key facts, with no false claim about what was asked.
- partial: some key facts right, but part missing (for example 3 of 6 names) or one wrong claim alongside right ones.
- wrong: the key facts are missing or false.
- no_answer: the answer declines or says the information is not available.

Extra true detail beyond the reference is fine. Treat Abram/Abraham, Sarai/Sarah and Jacob/Israel as the same person, and accept spelling variants of the same name. Grade only the facts, not style or citations. Give a one-sentence reason."""


class JudgeVerdict(BaseModel):
    verdict: Literal["correct", "partial", "wrong", "no_answer"]
    reason: str = Field(max_length=400)


def context_verses(
    trace: dict, passage_verses: dict[str, list[str]], statuses: set[str]
) -> set[str]:
    verses = set()
    for chunk in trace["chunks"]:
        if chunk["status"] in statuses:
            verses.update(passage_verses.get(chunk["id"], [chunk["id"]]))
    return verses


def score_trace(
    trace: dict, question: dict, passage_verses: dict[str, list[str]]
) -> dict:
    gold_verses = set(question["verses"])
    answer = trace["answer"] or {
        "sentences": [],
        "invalid_handles": [],
        "insufficient_context": True,
        "text": "",
    }
    has_retrieval = trace["query"]["mode"] != "closed_book"
    sent = context_verses(trace, passage_verses, {"context", "cited"})
    cited = context_verses(trace, passage_verses, {"cited"})
    valid_cites = sum(len(sentence["cites"]) for sentence in answer["sentences"])
    all_cites = valid_cites + len(answer["invalid_handles"])
    usage = trace["metrics"].get("usage", {})
    return {
        "qid": question["id"],
        "category": question["category"],
        "mode": trace["query"]["mode"],
        "answer": " ".join(sentence["text"] for sentence in answer["sentences"])
        or answer["text"],
        "verse_recall": len(gold_verses & sent) / len(gold_verses)
        if has_retrieval
        else None,
        "cited_verse_recall": len(gold_verses & cited) / len(gold_verses)
        if has_retrieval
        else None,
        "context_verses": len(sent),
        "invalid_citations": len(answer["invalid_handles"]),
        "citations": all_cites,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": usage.get("cost_usd", 0.0),
        "latency_ms": sum(trace["metrics"]["timings_ms"].values()),
        "failed": trace["answer"] is None,
    }


def judge_prompt(question: dict, answer: str) -> str:
    return f"Question: {question['question']}\n\nReference answer: {question['answer']}\n\nAnswer to grade: {answer or '(empty)'}"


def judge_rows(
    client: ClaudeCodeClient, rows: list[dict], questions: dict[str, dict], workers: int
) -> None:
    def judge(row: dict) -> None:
        result = client.call(
            f"{row['qid']}:{row['mode']}",
            JUDGE_SYSTEM,
            judge_prompt(questions[row["qid"]], row["answer"]),
            JudgeVerdict,
        )
        row["judge"] = result.output["verdict"]
        row["judge_reason"] = result.output["reason"]
        row["judge_cost_usd"] = 0.0 if result.cached else result.usage.cost_usd

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge, rows))


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(first == second for first, second in pairs) / len(pairs)
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)
    expected = (
        sum(first_counts[label] * second_counts[label] for label in first_counts)
        / len(pairs) ** 2
    )
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 4)


def mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(statistics.fmean(present), 4) if present else None


def summarize(rows: list[dict], human: dict[tuple[str, str], str]) -> dict:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    modes = {}
    for mode, mode_rows in by_mode.items():
        by_category: dict[str, list[float]] = defaultdict(list)
        for row in mode_rows:
            by_category[row["category"]].append(VERDICT_SCORE[row["judge"]])
        citations = sum(row["citations"] for row in mode_rows)
        modes[mode] = {
            "accuracy": mean_or_none(
                [VERDICT_SCORE[row["judge"]] for row in mode_rows]
            ),
            "verdicts": dict(Counter(row["judge"] for row in mode_rows)),
            "by_category": {
                category: round(statistics.fmean(scores), 4)
                for category, scores in sorted(by_category.items())
            },
            "verse_recall": mean_or_none([row["verse_recall"] for row in mode_rows]),
            "cited_verse_recall": mean_or_none(
                [row["cited_verse_recall"] for row in mode_rows]
            ),
            "invalid_citation_rate": round(
                sum(row["invalid_citations"] for row in mode_rows) / citations, 4
            )
            if citations
            else None,
            "mean_input_tokens": round(
                statistics.fmean(row["input_tokens"] for row in mode_rows)
            ),
            "mean_latency_s": round(
                statistics.fmean(row["latency_ms"] for row in mode_rows) / 1000, 2
            ),
            "cost_usd": round(sum(row["cost_usd"] for row in mode_rows), 4),
            "failed": sum(row["failed"] for row in mode_rows),
        }
    pairs = [
        (row["judge"], human[(row["qid"], row["mode"])])
        for row in rows
        if (row["qid"], row["mode"]) in human
    ]
    return {
        "questions": len({row["qid"] for row in rows}),
        "modes": modes,
        "judge_vs_human": {"graded": len(pairs), "kappa": cohen_kappa(pairs)},
    }


def read_human_grades(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            (row["qid"], row["mode"]): row["human"].strip()
            for row in csv.DictReader(handle)
            if row["human"].strip()
        }


def write_grades(
    path: Path,
    rows: list[dict],
    questions: dict[str, dict],
    human: dict[tuple[str, str], str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRADE_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (row["qid"], row["mode"])):
            writer.writerow(
                {
                    "qid": row["qid"],
                    "category": row["category"],
                    "mode": row["mode"],
                    "question": questions[row["qid"]]["question"],
                    "gold": questions[row["qid"]]["answer"],
                    "answer": row["answer"],
                    "judge": row["judge"],
                    "judge_reason": row["judge_reason"],
                    "human": human.get((row["qid"], row["mode"]), ""),
                }
            )


def load_run(run_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("*/*.json"))
    ]
