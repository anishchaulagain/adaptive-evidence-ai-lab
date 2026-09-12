"""Query slice: retrieve -> generate -> claim-level citations.

Real Postgres, real retrieval and real fusion; the embedder and chat model are
deterministic fakes so the assertions are about the pipeline rather than about
a model's mood.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from app.api.deps import get_embeddings, get_generator
from app.core.config import Settings
from app.db.session import get_sessionmaker
from app.services.embedding_service import EmbeddingService
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from core.reasoning.grounded import GroundedAnswerGenerator
from tests.fakes import FakeChatModel, FakeEmbeddingModel

pytestmark = pytest.mark.integration

QUERY = "/api/v1/query"

CORPUS = "\n\n".join(
    [
        "The NASA GISTEMP dataset records global surface temperature anomalies.",
        "Warming should be limited to 1.5 degrees above preindustrial levels.",
        "Coral reefs bleach when ocean temperatures stay elevated for weeks.",
        "Sovereign green bond issuance surpassed one trillion dollars.",
    ]
).encode()


@pytest.fixture
def fake_pipeline() -> BatchedEmbeddingPipeline:
    return BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=4)


@pytest.fixture
def chat_model() -> FakeChatModel:
    """Cites passage 0, which the corpus and probes make the right answer."""
    return FakeChatModel(
        {
            "answer": "Reefs bleach during prolonged ocean heat.",
            "claims": [
                {"text": "Bleaching follows sustained warmth.", "evidence": [0]},
            ],
        }
    )


@pytest.fixture
def app_with_fakes(
    app: FastAPI, fake_pipeline: BatchedEmbeddingPipeline, chat_model: FakeChatModel
) -> FastAPI:
    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(chat_model)

    app.dependency_overrides[get_embeddings] = _embeddings
    app.dependency_overrides[get_generator] = _generator
    return app


@pytest.fixture
async def corpus(
    client: AsyncClient, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> AsyncIterator[UUID]:
    created = await client.post("/api/v1/projects", json={"name": f"Query {uuid4().hex[:8]}"})
    project_id = UUID(created.json()["id"])

    from app.workers.tasks.ingestion import ingest_document

    response = await client.post(
        "/api/v1/documents/upload",
        params={"project_id": str(project_id)},
        files={"file": ("corpus.txt", CORPUS, "text/plain")},
    )
    document_id = UUID(response.json()["id"])
    await ingest_document({"settings": settings}, document_id)

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await EmbeddingService(session, settings, fake_pipeline).embed_document(document_id)

    yield project_id

    async with sessionmaker() as session:
        await session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": str(project_id)})
        await session.commit()


async def _ask(client: AsyncClient, project_id: UUID, **kwargs: object) -> dict[str, Any]:
    response = await client.post(
        QUERY,
        json={"project_id": str(project_id), "query": "why do reefs bleach", **kwargs},
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# --- answering ------------------------------------------------------------


async def test_query_returns_an_answer_with_claims(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    body = await _ask(client, corpus)

    assert body["answer"] == "Reefs bleach during prolonged ocean heat."
    assert body["abstained"] is False
    assert body["model"] == "fake-chat"
    assert body["strategy"] == "hybrid"
    assert len(body["claims"]) == 1
    assert body["claims"][0]["supported"] is True


async def test_citations_resolve_to_real_spans(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """A citation must point at an exact span of a real document, or the answer
    cannot be checked against its source."""
    body = await _ask(client, corpus)

    citations = body["claims"][0]["evidence"]
    assert citations
    for citation in citations:
        assert citation["chunk_id"]
        assert citation["document_id"]
        assert citation["char_start"] is not None
        assert citation["char_end"] is not None
        assert citation["text"]
        # The cited chunk must be one that was actually retrieved.
        assert citation["chunk_id"] in {item["chunk_id"] for item in body["evidence"]}


async def test_usage_and_latency_are_reported_per_stage(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """Retrieval and generation are billed and slow in different ways, so they
    are timed separately (spec section 25)."""
    body = await _ask(client, corpus)

    usage = body["usage"]
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 40
    assert usage["retrieval_latency_ms"] >= 0
    assert usage["generation_latency_ms"] >= 0
    assert usage["total_latency_ms"] == pytest.approx(
        usage["retrieval_latency_ms"] + usage["generation_latency_ms"], abs=0.02
    )


async def test_evidence_is_returned_by_default(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """An answer whose evidence cannot be inspected is not auditable."""
    body = await _ask(client, corpus)

    assert body["evidence"]
    assert body["evidence_count"] == len(body["evidence"])


async def test_evidence_can_be_omitted_from_the_payload(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    body = await _ask(client, corpus, include_evidence=False)

    assert body["evidence"] == []
    # The count still reports what was actually used to answer.
    assert body["evidence_count"] > 0


@pytest.mark.parametrize("strategy", ["hybrid", "semantic", "keyword"])
async def test_every_retrieval_strategy_can_answer(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID, strategy: str
) -> None:
    body = await _ask(client, corpus, query="coral reefs bleach", strategy=strategy)

    assert body["strategy"] == strategy
    assert body["answer"]
    # Only the vector strategies consume an embedding model.
    assert (body["embedding_model"] is not None) == (strategy != "keyword")


# --- groundedness signals -------------------------------------------------


async def test_invented_citations_are_reported(
    client: AsyncClient, app: FastAPI, corpus: UUID, fake_pipeline: BatchedEmbeddingPipeline
) -> None:
    """A model citing a passage that was never retrieved is a groundedness
    failure, and must surface in the response rather than be silently dropped."""
    model = FakeChatModel(
        {
            "answer": "An answer.",
            "claims": [{"text": "A claim.", "evidence": [0, 999]}],
        }
    )

    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(model)

    app.dependency_overrides[get_embeddings] = _embeddings
    app.dependency_overrides[get_generator] = _generator

    body = await _ask(client, corpus)

    assert body["invented_citations"] == 1
    assert len(body["claims"][0]["evidence"]) == 1


async def test_unsupported_claims_are_counted(
    client: AsyncClient, app: FastAPI, corpus: UUID, fake_pipeline: BatchedEmbeddingPipeline
) -> None:
    model = FakeChatModel(
        {
            "answer": "An answer.",
            "claims": [
                {"text": "Cited.", "evidence": [0]},
                {"text": "Uncited.", "evidence": []},
            ],
        }
    )

    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(model)

    app.dependency_overrides[get_embeddings] = _embeddings
    app.dependency_overrides[get_generator] = _generator

    body = await _ask(client, corpus)

    assert body["unsupported_claims"] == 1
    assert [claim["supported"] for claim in body["claims"]] == [True, False]


async def test_a_query_with_no_retrievable_evidence_abstains(
    client: AsyncClient, app: FastAPI, corpus: UUID, chat_model: FakeChatModel
) -> None:
    """Keyword retrieval is conjunctive, so a nonsense query retrieves nothing.
    Abstaining is the correct outcome, and the model must not be called."""

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(chat_model)

    app.dependency_overrides[get_generator] = _generator

    body = await _ask(client, corpus, query="zzzqqq nonexistent term", strategy="keyword")

    assert body["abstained"] is True
    assert body["evidence_count"] == 0
    assert body["claims"] == []
    assert chat_model.calls == []


# --- configuration --------------------------------------------------------


async def test_query_requires_a_provider_key(
    client: AsyncClient, app: FastAPI, corpus: UUID
) -> None:
    async def _no_generator() -> AsyncIterator[None]:
        yield None

    app.dependency_overrides[get_generator] = _no_generator
    response = await client.post(QUERY, json={"project_id": str(corpus), "query": "anything"})
    app.dependency_overrides.pop(get_generator, None)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PROVIDER_NOT_CONFIGURED"


async def test_query_is_scoped_to_the_project(client: AsyncClient, app_with_fakes: FastAPI) -> None:
    response = await client.post(QUERY, json={"project_id": str(uuid4()), "query": "anything"})

    assert response.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "no project"},
        {"project_id": "not-a-uuid", "query": "bad"},
        {"project_id": str(uuid4()), "query": ""},
        {"project_id": str(uuid4()), "query": "ok", "top_k": 0},
        {"project_id": str(uuid4()), "query": "ok", "strategy": "magic"},
    ],
    ids=["no-project", "bad-uuid", "empty-query", "top-k-zero", "bad-strategy"],
)
async def test_invalid_requests_are_rejected(
    client: AsyncClient, app_with_fakes: FastAPI, payload: dict[str, object]
) -> None:
    response = await client.post(QUERY, json=payload)

    assert response.status_code == 422
