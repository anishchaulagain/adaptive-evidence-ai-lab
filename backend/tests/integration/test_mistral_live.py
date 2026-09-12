"""Live Mistral checks.

Skipped unless MISTRAL_API_KEY is set, so the default suite stays offline,
deterministic and free. These are the assertions that a mocked transport
cannot make: that the real endpoint, model name and vector width are what the
schema assumes.

Run with:  pytest -m integration tests/integration/test_mistral_live.py
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.providers import get_embedding_pipeline
from app.models.chunk import EMBEDDING_DIMENSIONS

pytestmark = pytest.mark.integration


@pytest.fixture
def live_settings(settings: Settings) -> Settings:
    if settings.MISTRAL_API_KEY is None:
        pytest.skip("MISTRAL_API_KEY is not set; skipping live provider checks")
    return settings


async def test_live_embedding_matches_the_column_width(live_settings: Settings) -> None:
    """The real model must emit vectors the pgvector column can hold."""
    pipeline = get_embedding_pipeline(live_settings)
    try:
        vectors = await pipeline.embed_texts(["a short sentence", "another sentence"])
    finally:
        await pipeline.aclose()

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)


async def test_live_embeddings_are_unit_norm(live_settings: Settings) -> None:
    """Mistral documents its embeddings as norm 1. The HNSW index uses cosine
    ops, so this is not required for correctness — but if it ever stops holding,
    the assumption behind treating score as similarity deserves rechecking."""
    import math

    pipeline = get_embedding_pipeline(live_settings)
    try:
        vector = await pipeline.embed_query("a sentence to embed")
    finally:
        await pipeline.aclose()

    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0, abs=1e-3)


async def test_live_similar_text_scores_higher_than_unrelated(
    live_settings: Settings,
) -> None:
    """A smoke test that the model is semantically useful at all, independent
    of the retrieval plumbing."""
    pipeline = get_embedding_pipeline(live_settings)
    try:
        vectors = await pipeline.embed_texts(
            [
                "Greenhouse gas emissions must fall to limit global warming.",
                "Carbon emissions need to decline to slow climate change.",
                "The recipe calls for two tablespoons of olive oil.",
            ]
        )
    finally:
        await pipeline.aclose()

    def cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    related = cosine(vectors[0], vectors[1])
    unrelated = cosine(vectors[0], vectors[2])
    assert related > unrelated
