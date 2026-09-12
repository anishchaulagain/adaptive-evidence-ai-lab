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


@pytest.fixture
async def live_chat_settings(live_settings: Settings) -> Settings:
    """Settings for a key whose plan actually permits chat completions.

    Embeddings and chat are metered separately: a key can have embedding quota
    and zero chat quota, which the API reports as a 429 with
    `x-ratelimit-limit-req-minute: 0`. That is a plan limit, not a transient
    rate limit, so retrying cannot help and the check is skipped instead of
    reported as a product failure.
    """
    import httpx

    key = live_settings.MISTRAL_API_KEY
    assert key is not None
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{live_settings.MISTRAL_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key.get_secret_value()}"},
            json={
                "model": live_settings.GENERATION_MODEL,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ok"}],
            },
        )
    if response.status_code == 429:
        pytest.skip(
            "This key has no chat-completions quota "
            f"(limit={response.headers.get('x-ratelimit-limit-req-minute')}); "
            "skipping live generation checks"
        )
    return live_settings


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


# --- generation -----------------------------------------------------------


async def test_live_generation_returns_grounded_claims(live_chat_settings: Settings) -> None:
    """The real model must honour the JSON contract and cite by index."""
    from uuid import UUID, uuid4

    from app.core.providers import get_answer_generator
    from core.reasoning.base import InferenceBudget
    from core.types import Provenance, RetrievedChunk, RetrieverKind

    document_id = UUID("00000000-0000-0000-0000-0000000000aa")
    evidence = [
        RetrievedChunk(
            text="Coral reefs bleach when ocean temperatures stay elevated for weeks.",
            score=0.9,
            retriever=RetrieverKind.SEMANTIC,
            provenance=Provenance(document_id=document_id, chunk_id=uuid4()),
            rank=0,
        ),
        RetrievedChunk(
            text="Sovereign green bond issuance surpassed one trillion dollars.",
            score=0.5,
            retriever=RetrieverKind.SEMANTIC,
            provenance=Provenance(document_id=document_id, chunk_id=uuid4()),
            rank=1,
        ),
    ]

    generator = get_answer_generator(live_chat_settings)
    try:
        answer = await generator.generate(
            "Why do coral reefs bleach?",
            evidence,
            InferenceBudget(max_input_tokens=0, max_output_tokens=512),
        )
    finally:
        await generator.aclose()

    assert answer.text
    assert answer.abstained is False
    assert answer.claims, "the model must decompose the answer into claims"
    assert answer.usage.total > 0
    # Every citation must resolve to a passage that was actually supplied.
    valid = {chunk.provenance.chunk_id for chunk in evidence}
    assert set(answer.cited_chunk_ids) <= valid
    assert answer.metadata["invented_citations"] == 0
    # The relevant passage is the first one; the model should cite it.
    assert evidence[0].provenance.chunk_id in answer.cited_chunk_ids


async def test_live_model_abstains_when_evidence_is_irrelevant(
    live_chat_settings: Settings,
) -> None:
    """Refusing is the correct answer when the passages do not contain one.
    A model that invents an answer here would be the failure mode this whole
    platform exists to detect."""
    from uuid import UUID, uuid4

    from app.core.providers import get_answer_generator
    from core.reasoning.base import InferenceBudget
    from core.types import Provenance, RetrievedChunk, RetrieverKind

    document_id = UUID("00000000-0000-0000-0000-0000000000bb")
    evidence = [
        RetrievedChunk(
            text="Sovereign green bond issuance surpassed one trillion dollars.",
            score=0.4,
            retriever=RetrieverKind.SEMANTIC,
            provenance=Provenance(document_id=document_id, chunk_id=uuid4()),
            rank=0,
        )
    ]

    generator = get_answer_generator(live_chat_settings)
    try:
        answer = await generator.generate(
            "What is the boiling point of mercury?",
            evidence,
            InferenceBudget(max_input_tokens=0, max_output_tokens=512),
        )
    finally:
        await generator.aclose()

    assert answer.abstained is True, (
        f"expected abstention, got claims: {[c.text for c in answer.claims]}"
    )
