"""Adaptive retrieval end to end (spec sections 16, 76)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import text

from app.api.deps import get_embeddings
from app.core.config import Settings
from app.db.session import get_sessionmaker
from app.services.embedding_service import EmbeddingService
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from tests.fakes import FakeEmbeddingModel

pytestmark = pytest.mark.integration

SEARCH = "/api/v1/search"

CORPUS = "\n\n".join(
    [
        "Coral reefs bleach when ocean temperatures stay elevated for weeks.",
        "Deployment halted with error code ERR-5521 during the rollout phase.",
        "Deployment halted with error code ERR-9310 during the rollback phase.",
        "The NASA GISTEMP dataset records global surface temperature anomalies.",
        "Sovereign green bond issuance surpassed one trillion dollars.",
    ]
).encode()


@pytest.fixture
def fake_pipeline() -> BatchedEmbeddingPipeline:
    return BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=4)


@pytest.fixture
def app_with_fakes(app: FastAPI, fake_pipeline: BatchedEmbeddingPipeline) -> FastAPI:
    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    app.dependency_overrides[get_embeddings] = _embeddings
    return app


@pytest.fixture
async def corpus(
    client: AsyncClient, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> AsyncIterator[UUID]:
    created = await client.post("/api/v1/projects", json={"name": f"Adaptive {uuid4().hex[:8]}"})
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


async def _search(
    client: AsyncClient, project_id: UUID, query: str, strategy: str = "adaptive"
) -> dict[str, Any]:
    response = await client.post(
        SEARCH,
        json={
            "project_id": str(project_id),
            "query": query,
            "strategy": strategy,
            "top_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_adaptive_search_returns_results(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    body = await _search(client, corpus, "why do coral reefs bleach")

    assert body["strategy"] == "adaptive"
    assert body["results"]
    assert all(item["retriever"] == "hybrid" for item in body["results"])


async def test_the_response_explains_the_retrieval_choice(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """The platform's central question is "why was this strategy selected?".
    An adaptive result that cannot answer it is not useful."""
    body = await _search(client, corpus, "why do coral reefs bleach")

    analysis = body["analysis"]
    assert analysis is not None
    assert analysis["query_type"] == "conceptual"
    assert analysis["signals"]
    assert analysis["reason"]
    assert analysis["semantic_weight"] > analysis["keyword_weight"]


async def test_an_identifier_query_leans_lexical(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    body = await _search(client, corpus, "ERR-5521")

    analysis = body["analysis"]
    assert analysis["query_type"] == "exact_entity"
    assert "ERR-5521" in analysis["entities"]
    assert analysis["keyword_weight"] > analysis["semantic_weight"]


async def test_the_fixed_strategies_report_no_analysis(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """Only the strategy that makes a choice explains one."""
    for strategy in ("semantic", "keyword", "hybrid"):
        body = await _search(client, corpus, "coral reefs", strategy=strategy)
        assert body["analysis"] is None


async def test_adaptive_still_finds_the_exact_identifier(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """Weighting toward lexical must not lose the right passage."""
    body = await _search(client, corpus, "ERR-5521")

    assert "ERR-5521" in body["results"][0]["text"]


async def test_adaptive_carries_fusion_provenance(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """It fuses two arms, so each hit must still say which found it."""
    body = await _search(client, corpus, "ERR-5521")

    for item in body["results"]:
        assert item["retrieval_source"]
        assert item["fusion_score"] is not None


async def test_adaptive_needs_a_provider_key(
    client: AsyncClient, app: FastAPI, corpus: UUID
) -> None:
    async def _no_embeddings() -> AsyncIterator[None]:
        yield None

    app.dependency_overrides[get_embeddings] = _no_embeddings
    response = await client.post(
        SEARCH,
        json={"project_id": str(corpus), "query": "anything", "strategy": "adaptive"},
    )
    app.dependency_overrides.pop(get_embeddings, None)

    assert response.status_code == 503


async def test_an_adaptive_query_records_its_decision_on_the_trace(
    client: AsyncClient,
    app: FastAPI,
    corpus: UUID,
    fake_pipeline: BatchedEmbeddingPipeline,
) -> None:
    """Recorded on the trace so the choice is answerable from the stored
    execution, not only from a live call."""
    from app.api.deps import get_generator
    from core.reasoning.grounded import GroundedAnswerGenerator
    from tests.fakes import FakeChatModel

    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(FakeChatModel({"answer": "An answer.", "claims": []}))

    app.dependency_overrides[get_embeddings] = _embeddings
    app.dependency_overrides[get_generator] = _generator

    answer = await client.post(
        "/api/v1/query",
        json={"project_id": str(corpus), "query": "ERR-5521", "strategy": "adaptive"},
    )
    assert answer.status_code == 200, answer.text

    trace = (await client.get(f"/api/v1/traces/{answer.json()['trace_id']}")).json()
    retrieval = next(s for s in trace["spans"] if s["stage"] == "retrieval")

    assert retrieval["attributes"]["query_type"] == "exact_entity"
    assert retrieval["attributes"]["keyword_weight"] > 0.5
    assert retrieval["attributes"]["signals"]


async def test_an_adaptive_evaluation_run_completes(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID, settings: Settings
) -> None:
    """Adaptive must be measurable the same way the fixed strategies are —
    that comparison is the point of the phase."""
    dataset = await client.post(
        "/api/v1/evaluations/datasets",
        json={
            "project_id": str(corpus),
            "name": f"Adaptive {uuid4().hex[:8]}",
            "items": [{"question": "coral reefs bleach"}, {"question": "ERR-5521"}],
        },
    )
    created = await client.post(
        "/api/v1/evaluations",
        json={"dataset_id": dataset.json()["id"], "strategy": "adaptive"},
    )

    from app.services.evaluation_service import EvaluationService

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await EvaluationService(session, settings).execute(UUID(created.json()["id"]))

    run = (await client.get(f"/api/v1/evaluations/{created.json()['id']}")).json()
    assert run["status"] == "completed"
    assert run["config"]["strategy"] == "adaptive"
