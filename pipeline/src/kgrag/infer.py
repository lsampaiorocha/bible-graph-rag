from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from kgrag.build import edge_id
from kgrag.llm import LLMClient

PROMPT_VERSION = "confirm_parent_v1"
PURPOSE = "parent_confirm"
PARENT_TYPES = ("FATHER_OF", "MOTHER_OF")
MAX_ROUNDS = 3
MAX_VERSES = 10


@dataclass(frozen=True)
class ParentCandidate:
    edge_type: str
    parent: str
    child: str
    rule: str
    via: str
    verses: tuple[str, ...]
    passages: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.edge_type}|{self.parent}|{self.child}"


ParentConfirmer = Callable[[ParentCandidate], tuple[bool, str]]


class ParentVerdict(BaseModel):
    verdict: Literal["yes", "no", "unsure"]
    reason: str = Field(description="One sentence naming the evidence.")


def propose_parent_links(
    entities: list[dict], edges: list[dict]
) -> list[ParentCandidate]:
    # Verses often state one parent and leave the other to context ("Adah bare Jabal",
    # "his wife bare a son"), so a parent link can follow from links already in the graph.
    gender = {
        entity["id"]: entity.get("gender")
        for entity in entities
        if entity["kind"] == "person"
    }
    parent_edges: dict[str, dict[str, list[dict]]] = {
        edge_type: defaultdict(list) for edge_type in PARENT_TYPES
    }
    partner_edges: dict[str, dict[str, list[dict]]] = {
        "SPOUSE_OF": defaultdict(list),
        "SIBLING_OF": defaultdict(list),
    }
    for edge in edges:
        if edge["type"] in parent_edges:
            parent_edges[edge["type"]][edge["target"]].append(edge)
        elif edge["type"] in partner_edges:
            partner_edges[edge["type"]][edge["source"]].append(edge)
            partner_edges[edge["type"]][edge["target"]].append(edge)

    def other(edge: dict, person: str) -> str:
        return edge["target"] if edge["source"] == person else edge["source"]

    def candidate(
        edge_type: str, parent: str, child: str, rule: str, via: str, *basis: dict
    ) -> ParentCandidate:
        return ParentCandidate(
            edge_type,
            parent,
            child,
            rule,
            via,
            tuple(
                dict.fromkeys(
                    verse for edge in basis for verse in edge["evidence_verses"]
                )
            ),
            tuple(
                sorted(
                    {passage for edge in basis for passage in edge["evidence_passages"]}
                )
            ),
        )

    found: dict[str, ParentCandidate] = {}
    for child in sorted(gender):
        for missing, known, partner_gender, rule in (
            ("FATHER_OF", "MOTHER_OF", "male", "the mother's only husband"),
            ("MOTHER_OF", "FATHER_OF", "female", "the father's only wife"),
        ):
            if parent_edges[missing][child]:
                continue
            for known_edge in parent_edges[known][child]:
                known_parent = known_edge["source"]
                partners = [
                    edge
                    for edge in partner_edges["SPOUSE_OF"][known_parent]
                    if gender.get(other(edge, known_parent)) == partner_gender
                ]
                if len(partners) == 1:
                    proposal = candidate(
                        missing,
                        other(partners[0], known_parent),
                        child,
                        rule,
                        known_parent,
                        known_edge,
                        partners[0],
                    )
                    found.setdefault(proposal.key, proposal)
        if parent_edges["FATHER_OF"][child] or parent_edges["MOTHER_OF"][child]:
            continue
        for sibling_edge in partner_edges["SIBLING_OF"][child]:
            sibling = other(sibling_edge, child)
            for edge_type in PARENT_TYPES:
                for parent_edge in parent_edges[edge_type][sibling]:
                    proposal = candidate(
                        edge_type,
                        parent_edge["source"],
                        child,
                        "a sibling's parent",
                        sibling,
                        sibling_edge,
                        parent_edge,
                    )
                    found.setdefault(proposal.key, proposal)
    return sorted(found.values(), key=lambda proposal: proposal.key)


def infer_parent_links(
    entities: list[dict],
    edges: list[dict],
    verse_order: dict[str, int],
    confirm: ParentConfirmer,
) -> tuple[list[dict], list[dict]]:
    # Rounds let one confirmed link enable the next: once Jabal has a father, his
    # parentless brother Jubal can take both of Jabal's parents.
    added: list[dict] = []
    decisions: list[dict] = []
    asked: set[str] = set()
    for _ in range(MAX_ROUNDS):
        fresh = [
            proposal
            for proposal in propose_parent_links(entities, edges + added)
            if proposal.key not in asked
        ]
        if not fresh:
            break
        for proposal in fresh:
            asked.add(proposal.key)
            confirmed, reason = confirm(proposal)
            decisions.append(
                {**asdict(proposal), "confirmed": confirmed, "reason": reason}
            )
            if confirmed:
                added.append(
                    {
                        "id": edge_id(
                            proposal.edge_type, proposal.parent, proposal.child, None
                        ),
                        "type": proposal.edge_type,
                        "source": proposal.parent,
                        "target": proposal.child,
                        "other_label": None,
                        "description": reason,
                        "evidence_verses": sorted(
                            proposal.verses, key=verse_order.__getitem__
                        ),
                        "evidence_passages": list(proposal.passages),
                        "support_count": len(proposal.passages),
                        "inferred": True,
                    }
                )
    return added, decisions


def load_prompt() -> str:
    return (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(
        encoding="utf-8"
    )


def confirmation_prompt(
    proposal: ParentCandidate, names: dict[str, str], verse_text: dict[str, str]
) -> str:
    role = "father" if proposal.edge_type == "FATHER_OF" else "mother"
    basis = {
        "the mother's only husband": f"{names[proposal.via]} is {names[proposal.child]}'s mother, and {names[proposal.parent]} is her only husband in the graph.",
        "the father's only wife": f"{names[proposal.via]} is {names[proposal.child]}'s father, and {names[proposal.parent]} is his only wife in the graph.",
        "a sibling's parent": f"{names[proposal.child]} is a sibling of {names[proposal.via]}, whose {role} is {names[proposal.parent]}.",
    }[proposal.rule]
    return "\n".join(
        [
            f"Proposed link: {names[proposal.parent]} is the {role} of {names[proposal.child]}.",
            f"Why it was proposed: {basis}",
            "Verses:",
            *(
                f"[{verse}] {verse_text[verse]}"
                for verse in proposal.verses[:MAX_VERSES]
            ),
        ]
    )


def llm_parent_confirmer(
    client: LLMClient, names: dict[str, str], verse_text: dict[str, str]
) -> ParentConfirmer:
    system_prompt = load_prompt()

    def confirm(proposal: ParentCandidate) -> tuple[bool, str]:
        result = client.call(
            proposal.key,
            system_prompt,
            confirmation_prompt(proposal, names, verse_text),
            ParentVerdict,
        )
        return result.output["verdict"] == "yes", result.output["reason"]

    return confirm
