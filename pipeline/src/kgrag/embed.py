import re
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VERSE_MARKER = re.compile(r"^\[\S+\] ", re.MULTILINE)


def entity_profile(entity: dict) -> str:
    aliases = (
        f" Also called {', '.join(entity['aliases'])}." if entity["aliases"] else ""
    )
    return (
        f"{entity['name']} ({entity['kind']}).{aliases} {entity['description']}".strip()
    )


def passage_body(passage_text: str) -> str:
    return VERSE_MARKER.sub("", passage_text)


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The browser receives int8 vectors; Neo4j gets the same values dequantised, so both
    # sides rank with identical numbers.
    scales = np.abs(vectors).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    return np.round(vectors / scales[:, None]).astype(np.int8), scales.astype(
        np.float32
    )


def dequantize(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float32) * scales[:, None]


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return model.encode(
        texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True
    )


def save_embeddings(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    quantized, scales = quantize(vectors)
    np.savez(path, ids=np.array(ids), quantized=quantized, scales=scales)


def load_embeddings(path: Path) -> dict[str, list[float]]:
    data = np.load(path)
    vectors = dequantize(data["quantized"], data["scales"])
    return {
        str(item_id): vector.tolist()
        for item_id, vector in zip(data["ids"], vectors, strict=True)
    }
