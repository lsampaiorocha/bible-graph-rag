from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from kgrag.llm import ClaudeCodeClient
from kgrag.resolve import KIN_RELATIONS, Cluster, Occurrence

PROMPT_VERSION = "confirm_same_entity_v1"
PURPOSE = "er_confirm"
MAX_VERSES = 3
MAX_DESCRIPTIONS = 3


class SameEntityVerdict(BaseModel):
    verdict: Literal["same", "different", "unsure"]
    reason: str = Field(description="One sentence naming the evidence.")


def load_prompt() -> str:
    return (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(
        encoding="utf-8"
    )


def describe(
    names: set[str],
    descriptions: list[str],
    kin: dict[str, set[str]],
    verses: list[str],
    verse_text: dict[str, str],
) -> str:
    relatives = "; ".join(
        f"{relation}: {', '.join(sorted(kin[relation]))}"
        for relation in KIN_RELATIONS
        if kin[relation]
    )
    lines = [
        f"Names: {', '.join(sorted(names))}",
        *(
            f"Description: {description}"
            for description in descriptions[:MAX_DESCRIPTIONS]
        ),
        f"Relatives: {relatives or 'none named'}",
        *(f"[{verse}] {verse_text[verse]}" for verse in verses[:MAX_VERSES]),
    ]
    return "\n".join(lines)


def confirmation_prompt(
    occurrence: Occurrence, cluster: Cluster, verse_text: dict[str, str]
) -> str:
    cluster_kin = {relation: cluster.kin(relation) for relation in KIN_RELATIONS}
    cluster_verses = [verse for other in cluster.occurrences for verse in other.verses]
    descriptions = list(
        dict.fromkeys(
            other.description for other in cluster.occurrences if other.description
        )
    )
    return (
        "Mention A:\n"
        + describe(
            occurrence.names,
            [occurrence.description],
            occurrence.kin,
            occurrence.verses,
            verse_text,
        )
        + "\n\nEntity B:\n"
        + describe(cluster.names, descriptions, cluster_kin, cluster_verses, verse_text)
    )


def llm_confirmer(client: ClaudeCodeClient, verse_text: dict[str, str]):
    system_prompt = load_prompt()

    def confirm(occurrence: Occurrence, cluster: Cluster) -> bool:
        item_id = f"{occurrence.passage_id}:{occurrence.key[2]}"
        result = client.call(
            item_id,
            system_prompt,
            confirmation_prompt(occurrence, cluster, verse_text),
            SameEntityVerdict,
        )
        # "unsure" counts as different: a wrong merge corrupts family lines, a wrong split
        # only leaves one person in two nodes.
        return result.output["verdict"] == "same"

    return confirm
