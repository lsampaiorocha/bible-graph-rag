from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from kgrag.build import edge_id
from kgrag.llm import ClaudeCodeClient

PROMPT_VERSION = "check_kinship_word_v1"
PURPOSE = "kinship_check"
MAX_VERSES = 10
FOREFATHER = "a second father who descends from this one"
KINSMAN = "a sibling whose parent is the other's sibling"
LOOSE_LABEL = {FOREFATHER: "forefather of", KINSMAN: "kinsman of"}


@dataclass(frozen=True)
class KinshipFlag:
    edge_id: str
    edge_type: str
    source: str
    target: str
    rule: str
    via: str
    verses: tuple[str, ...]


KinshipChecker = Callable[[KinshipFlag], tuple[str, str]]


class KinshipVerdict(BaseModel):
    verdict: Literal["literal", "loose", "unsure"]
    reason: str = Field(description="One sentence naming the evidence.")


def flag_loose_kinship(edges: list[dict]) -> list[KinshipFlag]:
    # The KJV says "son" for a grandson and "brother" for a nephew or kinsman. Read
    # literally, that gives a person two fathers, one descended from the other, or a
    # "sibling" whose parent is the other's sibling.
    parent_edges: dict[str, list[dict]] = defaultdict(list)
    children: dict[str, set[str]] = defaultdict(set)
    sibling_edges: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        if edge["type"] in ("FATHER_OF", "MOTHER_OF"):
            parent_edges[edge["target"]].append(edge)
            children[edge["source"]].add(edge["target"])
        elif edge["type"] == "SIBLING_OF":
            sibling_edges[edge["source"]].append(edge)
            sibling_edges[edge["target"]].append(edge)

    def ancestor_path(
        person: str, ancestor: str, seen: frozenset[str] = frozenset()
    ) -> list[dict] | None:
        for edge in parent_edges.get(person, []):
            if edge["source"] == ancestor:
                return [edge]
            if edge["source"] not in seen:
                rest = ancestor_path(edge["source"], ancestor, seen | {person})
                if rest is not None:
                    return [edge, *rest]
        return None

    def sibling_basis(person: str, other: str) -> list[dict] | None:
        for edge in sibling_edges.get(person, []):
            if other in (edge["source"], edge["target"]):
                return [edge]
        for edge in parent_edges.get(person, []):
            for shared in parent_edges.get(other, []):
                if shared["source"] == edge["source"]:
                    return [edge, shared]
        return None

    def verses_of(*basis: dict) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(verse for edge in basis for verse in edge["evidence_verses"])
        )

    flags: dict[str, KinshipFlag] = {}
    for child, incoming in parent_edges.items():
        fathers = [edge for edge in incoming if edge["type"] == "FATHER_OF"]
        for flagged in fathers:
            for other in fathers:
                path = (
                    None
                    if other is flagged
                    else ancestor_path(other["source"], flagged["source"])
                )
                if path is not None:
                    flags.setdefault(
                        flagged["id"],
                        KinshipFlag(
                            flagged["id"],
                            "FATHER_OF",
                            flagged["source"],
                            child,
                            FOREFATHER,
                            other["source"],
                            verses_of(flagged, other, *path),
                        ),
                    )
    for edge in edges:
        if edge["type"] != "SIBLING_OF":
            continue
        for elder, younger in (
            (edge["source"], edge["target"]),
            (edge["target"], edge["source"]),
        ):
            for parent_edge in parent_edges.get(younger, []):
                basis = sibling_basis(parent_edge["source"], elder)
                if basis is not None:
                    flags.setdefault(
                        edge["id"],
                        KinshipFlag(
                            edge["id"],
                            "SIBLING_OF",
                            elder,
                            younger,
                            KINSMAN,
                            parent_edge["source"],
                            verses_of(edge, parent_edge, *basis),
                        ),
                    )
    return sorted(flags.values(), key=lambda flag: flag.edge_id)


def relabel_loose_kinship(
    edges: list[dict], check: KinshipChecker
) -> tuple[list[dict], list[dict]]:
    decisions = []
    relabelled: dict[str, dict] = {}
    for flag in flag_loose_kinship(edges):
        verdict, reason = check(flag)
        decisions.append({**asdict(flag), "verdict": verdict, "reason": reason})
        if verdict == "loose":
            relabelled[flag.edge_id] = {
                "rule": flag.rule,
                "source": flag.source,
                "target": flag.target,
                "reason": reason,
            }
    result = []
    for edge in edges:
        change = relabelled.get(edge["id"])
        if change is None:
            result.append(edge)
            continue
        label = LOOSE_LABEL[change["rule"]]
        result.append(
            {
                **edge,
                "id": edge_id("OTHER", change["source"], change["target"], label),
                "type": "OTHER",
                "source": change["source"],
                "target": change["target"],
                "other_label": label,
                "description": change["reason"],
            }
        )
    return sorted(result, key=lambda edge: edge["id"]), decisions


def load_prompt() -> str:
    return (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(
        encoding="utf-8"
    )


def check_prompt(
    flag: KinshipFlag, names: dict[str, str], verse_text: dict[str, str]
) -> str:
    if flag.rule == FOREFATHER:
        link = f"{names[flag.source]} is the father of {names[flag.target]}."
        why = f"{names[flag.target]}'s other father in the graph, {names[flag.via]}, descends from {names[flag.source]}."
    else:
        link = f"{names[flag.source]} and {names[flag.target]} are siblings."
        why = f"{names[flag.target]}'s parent in the graph, {names[flag.via]}, is a sibling of {names[flag.source]}."
    return "\n".join(
        [
            f"Link in the graph: {link}",
            f"Why it was flagged: {why}",
            "Verses:",
            *(f"[{verse}] {verse_text[verse]}" for verse in flag.verses[:MAX_VERSES]),
        ]
    )


def llm_kinship_checker(
    client: ClaudeCodeClient, names: dict[str, str], verse_text: dict[str, str]
) -> KinshipChecker:
    system_prompt = load_prompt()

    def check(flag: KinshipFlag) -> tuple[str, str]:
        result = client.call(
            flag.edge_id,
            system_prompt,
            check_prompt(flag, names, verse_text),
            KinshipVerdict,
        )
        return result.output["verdict"], result.output["reason"]

    return check
