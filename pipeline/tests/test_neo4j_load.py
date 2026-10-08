import numpy as np

from kgrag.embed import dequantize, quantize
from kgrag.neo4j_load import alias_key


def test_alias_key_normalizes_titles_hyphens_and_possessives_like_question_ngrams():
    assert [
        alias_key(name)
        for name in ["the LORD", "Beth-el", "Rebekah’s", "Lord GOD", "  Padan-aram "]
    ] == [
        "lord",
        "bethel",
        "rebekah",
        "lord god",
        "padanaram",
    ]


def test_int8_round_trip_keeps_vectors_nearly_identical():
    vectors = np.array([[0.6, -0.8, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    restored = dequantize(*quantize(vectors))

    assert np.allclose(restored, vectors, atol=0.01)
