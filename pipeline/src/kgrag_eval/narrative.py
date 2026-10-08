import csv
import math
import random
from collections import Counter
from pathlib import Path

from kgrag.schema import NARRATIVE_TYPES

LABELS = ("correct", "wrong_type", "wrong_direction", "unsupported")
COLUMNS = (
    "edge_id",
    "type",
    "other_label",
    "source",
    "target",
    "description",
    "verses",
    "verse_text",
    "label",
    "note",
)


def sample_narrative_edges(
    entities: list[dict],
    edges: list[dict],
    verse_text: dict[str, str],
    size: int,
    seed: int,
) -> list[dict]:
    names = {entity["id"]: entity["name"] for entity in entities}
    narrative = sorted(
        (edge for edge in edges if edge["type"] in NARRATIVE_TYPES),
        key=lambda edge: edge["id"],
    )
    chosen = random.Random(seed).sample(narrative, min(size, len(narrative)))
    return [
        {
            "edge_id": edge["id"],
            "type": edge["type"],
            "other_label": edge.get("other_label") or "",
            "source": names[edge["source"]],
            "target": names[edge["target"]],
            "description": edge["description"],
            "verses": " ".join(edge["evidence_verses"]),
            "verse_text": " | ".join(
                f"{ref}: {verse_text[ref]}"
                for ref in edge["evidence_verses"]
                if ref in verse_text
            ),
            "label": "",
            "note": "",
        }
        for edge in sorted(chosen, key=lambda edge: edge["id"])
    ]


def write_sample(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_labels(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    # Unlike p ± z·sqrt(p(1-p)/n), stays inside 0..1 and behaves at small n or p near 1.
    if total == 0:
        return 0.0, 0.0
    share = successes / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    margin = (
        z
        * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def score_labels(rows: list[dict]) -> dict | None:
    labels = Counter(row["label"].strip() for row in rows if row["label"].strip())
    unknown = set(labels) - set(LABELS)
    if unknown:
        raise ValueError(
            f"unknown labels {sorted(unknown)}; use one of {', '.join(LABELS)}"
        )
    labeled = sum(labels.values())
    if labeled == 0:
        return None
    low, high = wilson_interval(labels["correct"], labeled)
    return {
        "labeled": labeled,
        "sampled": len(rows),
        "precision": labels["correct"] / labeled,
        "interval_95": [low, high],
        "by_label": {label: labels[label] for label in LABELS},
    }
