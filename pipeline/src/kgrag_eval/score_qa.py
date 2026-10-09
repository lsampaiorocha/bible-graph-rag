import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from kgrag.llm import LLMClient
from kgrag_eval.graph_score import gold_family_triples
from kgrag_eval.theographic import Gold

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


SHOWN = {"context", "cited"}
PER_QUESTION_METRICS = [
    "key_people_shown",
    "answer_name_recall",
    "tree_people_recall",
    "tree_people_precision",
    "tree_link_recall",
    "tree_link_precision",
]
PARENT_TYPES = {"FATHER_OF", "MOTHER_OF"}

GROUNDING_SYSTEM = """You check whether sources back each sentence of an answer about Genesis (King James Version).

For each numbered sentence, answer supported when the sources listed under it state or directly imply every claim in the sentence, and unsupported otherwise. A source line starting [entity] or [link] is a summary the graph extracted from the verses; a [link] line lists those verses under it. Use only the sources given, never outside knowledge."""


class SentenceSupport(BaseModel):
    index: int
    supported: bool


class GroundingVerdict(BaseModel):
    sentences: list[SentenceSupport]


def lineage_gold(
    gold: Gold, person: str, direction: str, scope: str
) -> tuple[set[str], set[tuple[str, str]]]:
    triples = gold_family_triples(gold, scope)
    links = {
        (parent, child)
        for _, parent, child in triples["FATHER_OF"] | triples["MOTHER_OF"]
    }
    next_of: dict[str, set[str]] = defaultdict(set)
    for parent, child in links:
        if direction == "ancestors":
            next_of[child].add(parent)
        else:
            next_of[parent].add(child)
    found: set[str] = set()
    frontier = {person}
    while frontier:
        frontier = {
            relative for current in frontier for relative in next_of[current]
        } - (found | {person})
        found |= frontier
    members = found | {person}
    return found, {
        (parent, child)
        for parent, child in links
        if parent in members and child in members
    }


def share(part: set, whole: set) -> float | None:
    return len(part) / len(whole) if whole else None


def score_lineage(
    trace: dict,
    answer_text: str,
    root: str,
    people: set[str],
    links: set[tuple[str, str]],
    gold_of_entity: dict[str, str],
    names_of: dict[str, set[str]],
) -> dict:
    named = {
        slug
        for slug in people
        if any(
            re.search(rf"\b{re.escape(name)}\b", answer_text) for name in names_of[slug]
        )
    }
    scores = {"answer_name_recall": share(named, people)}
    if trace["query"]["mode"] != "agent":
        return scores | dict.fromkeys(
            [
                "tree_people_recall",
                "tree_people_precision",
                "tree_link_recall",
                "tree_link_precision",
            ]
        )

    def gold_id(entity_id: str) -> str:
        return gold_of_entity.get(entity_id, f"unmatched:{entity_id}")

    shown_people = {
        gold_id(node["id"])
        for node in trace["nodes"]
        if node["status"] in SHOWN and node["id"].startswith("per_")
    } - {root}
    shown_links = {
        (gold_id(edge["source"]), gold_id(edge["target"]))
        for edge in trace["edges"]
        if edge["status"] in SHOWN and edge["type"] in PARENT_TYPES
    }
    return scores | {
        "tree_people_recall": share(shown_people & people, people),
        "tree_people_precision": share(shown_people & people, shown_people),
        "tree_link_recall": share(shown_links & links, links),
        "tree_link_precision": share(shown_links & links, shown_links),
    }


def key_people_shown(
    trace: dict, people: list[str], gold_of_entity: dict[str, str]
) -> float | None:
    if trace["query"]["mode"] != "agent":
        return None
    shown = {
        gold_of_entity.get(node["id"])
        for node in trace["nodes"]
        if node["status"] in SHOWN
    }
    return share(set(people) & shown, set(people))


def source_lines(
    cite: str,
    trace: dict,
    verse_text: dict[str, str],
    passage_text: dict[str, str],
    card_text: dict[str, str],
) -> list[str]:
    if cite in passage_text:
        return [f"[{cite}] {passage_text[cite]}"]
    if cite in verse_text:
        return [f"[{cite}] {verse_text[cite]}"]
    edge = next((edge for edge in trace["edges"] if edge["id"] == cite), None)
    if edge is None:
        return [f"[entity] {card_text.get(cite, cite)}"]
    return [
        f"[link] {card_text.get(cite, cite)}",
        *(
            f"  [{verse}] {verse_text[verse]}"
            for verse in edge["evidence_chunk_ids"]
            if verse in verse_text
        ),
    ]


def check_grounding(
    client: LLMClient,
    pairs: list[tuple[dict, dict]],
    verse_text: dict[str, str],
    passage_text: dict[str, str],
    card_text: dict[str, str],
    workers: int,
) -> None:
    # Uncited sentences count as unsupported without asking; so does every
    # sentence of an answer given from memory.
    def check(pair: tuple[dict, dict]) -> None:
        row, trace = pair
        sentences = (trace["answer"] or {}).get("sentences", [])
        row["sentences"] = len(sentences)
        row["grounded_sentences"] = 0
        cited = [
            (index, sentence)
            for index, sentence in enumerate(sentences)
            if sentence["cites"]
        ]
        if not cited:
            return
        prompt = "\n\n".join(
            f"{index}. {sentence['text']}\nSources:\n"
            + "\n".join(
                line
                for cite in sentence["cites"]
                for line in source_lines(
                    cite, trace, verse_text, passage_text, card_text
                )
            )
            for index, sentence in cited
        )
        result = client.call(
            f"{row['qid']}:{row['mode']}:grounding",
            GROUNDING_SYSTEM,
            prompt,
            GroundingVerdict,
        )
        asked = {index for index, _ in cited}
        row["grounded_sentences"] = sum(
            verdict["supported"]
            for verdict in result.output["sentences"]
            if verdict["index"] in asked
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check, pairs))


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
    gold_verses = set(question.get("verses", []))
    answer = trace["answer"] or {
        "sentences": [],
        "invalid_handles": [],
        "insufficient_context": True,
        "text": "",
    }
    has_retrieval = trace["query"]["mode"] != "closed_book" and bool(gold_verses)
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
    client: LLMClient, rows: list[dict], questions: dict[str, dict], workers: int
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
            "grounded_rate": round(
                sum(row["grounded_sentences"] for row in mode_rows)
                / max(1, sum(row["sentences"] for row in mode_rows)),
                4,
            ),
            **{
                metric: mean_or_none([row.get(metric) for row in mode_rows])
                for metric in PER_QUESTION_METRICS
            },
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
