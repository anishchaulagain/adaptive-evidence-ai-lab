"""Trace persistence and retrieval (spec sections 23, 73).

Every query must leave an inspectable record — including a query that failed,
which is the one most worth inspecting.
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
from core.errors import ErrorCode, ProviderError
from core.reasoning.grounded import GroundedAnswerGenerator
from tests.fakes import FakeChatModel, FakeEmbeddingModel

pytestmark = pytest.mark.integration

QUERY = "/api/v1/query"
TRACES = "/api/v1/traces"

CORPUS = "\n\n".join(
    [
        "Coral reefs bleach when ocean temperatures stay elevated for weeks.",
        "The NASA GISTEMP dataset records global surface temperature anomalies.",
        "Sovereign green bond issuance surpassed one trillion dollars.",
    ]
).encode()


class ExplodingChatModel:
    """A chat model that always fails, to exercise the failure trace path."""

    model_id = "exploding-chat"

    async def complete_json(self, **kwargs: Any) -> Any:
        raise ProviderError(
            "the model is down",
            code=ErrorCode.MODEL_UNAVAILABLE,
            provider="mistral",
            model=self.model_id,
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_pipeline() -> BatchedEmbeddingPipeline:
    return BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=4)


def _install(app: FastAPI, pipeline: BatchedEmbeddingPipeline, model: Any) -> None:
    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield pipeline

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(model)

    app.dependency_overrides[get_embeddings] = _embeddings
    app.dependency_overrides[get_generator] = _generator


@pytest.fixture
def app_with_fakes(app: FastAPI, fake_pipeline: BatchedEmbeddingPipeline) -> FastAPI:
    _install(
        app,
        fake_pipeline,
        FakeChatModel(
            {
                "answer": "Reefs bleach during prolonged ocean heat.",
                "claims": [{"text": "Heat drives bleaching.", "evidence": [0]}],
            }
        ),
    )
    return app


@pytest.fixture
async def corpus(
    client: AsyncClient, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> AsyncIterator[UUID]:
    created = await client.post("/api/v1/projects", json={"name": f"Traces {uuid4().hex[:8]}"})
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


async def _trace(client: AsyncClient, trace_id: str) -> dict[str, Any]:
    response = await client.get(f"{TRACES}/{trace_id}")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# --- every query is traced ------------------------------------------------


async def test_a_query_returns_a_trace_id_that_resolves(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    answer = await _ask(client, corpus)

    assert answer["trace_id"]
    trace = await _trace(client, answer["trace_id"])
    assert trace["status"] == "ok"
    assert trace["query_text"] == "why do reefs bleach"
    assert trace["strategy"] == "hybrid"


async def test_the_trace_contains_a_span_per_stage(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    answer = await _ask(client, corpus)

    trace = await _trace(client, answer["trace_id"])
    stages = [span["stage"] for span in trace["spans"]]
    assert "retrieval" in stages
    assert "generation" in stages
    assert all(span["status"] == "ok" for span in trace["spans"])


async def test_spans_carry_their_stage_metadata(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """A span with no metadata records that a stage ran, which is the least
    useful thing about it."""
    answer = await _ask(client, corpus)

    trace = await _trace(client, answer["trace_id"])
    by_stage = {span["stage"]: span for span in trace["spans"]}
    assert by_stage["retrieval"]["attributes"]["candidate_count"] > 0
    assert by_stage["generation"]["attributes"]["model"] == "fake-chat"
    assert by_stage["generation"]["attributes"]["claims"] == 1


async def test_spans_are_ordered_and_offset_for_a_timeline(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """Spec section 24 draws an execution timeline; that needs each span's
    offset from the trace start, in execution order."""
    answer = await _ask(client, corpus)

    spans = (await _trace(client, answer["trace_id"]))["spans"]
    offsets = [span["offset_ms"] for span in spans]
    assert offsets == sorted(offsets)
    assert all(span["offset_ms"] >= 0 for span in spans)
    assert all(span["duration_ms"] >= 0 for span in spans)


# --- metrics (spec section 25) --------------------------------------------


async def test_the_trace_reports_tokens_and_per_stage_latency(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    answer = await _ask(client, corpus)

    metrics = (await _trace(client, answer["trace_id"]))["metrics"]
    assert metrics["input_tokens"] == 100
    assert metrics["output_tokens"] == 40
    assert metrics["total_tokens"] == 140
    assert metrics["retrieval_latency_ms"] is not None
    assert metrics["generation_latency_ms"] is not None
    assert metrics["total_latency_ms"] >= metrics["generation_latency_ms"]
    assert metrics["evidence_count"] > 0
    # No reranking or verification stage exists yet, so those must be absent
    # rather than reported as zero.
    assert metrics["reranking_latency_ms"] is None
    assert metrics["verification_latency_ms"] is None


async def test_cost_is_null_without_configured_pricing(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """A fabricated zero would understate what a benchmark run cost."""
    answer = await _ask(client, corpus)

    assert answer["usage"]["cost"] is None
    assert (await _trace(client, answer["trace_id"]))["metrics"]["cost"] is None


async def test_the_model_and_provider_are_recorded(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    answer = await _ask(client, corpus)

    trace = await _trace(client, answer["trace_id"])
    assert trace["model"] == "fake-chat"
    assert trace["provider"] == "mistral"
    assert trace["embedding_model"] == "fake-embed"


# --- failures are traced too ----------------------------------------------


async def test_a_failed_query_still_persists_its_trace(
    client: AsyncClient, app: FastAPI, corpus: UUID, fake_pipeline: BatchedEmbeddingPipeline
) -> None:
    """Spec section 47: errors must appear in traces. A failure that leaves no
    record is the hardest kind to diagnose."""
    _install(app, fake_pipeline, ExplodingChatModel())

    response = await client.post(
        QUERY, json={"project_id": str(corpus), "query": "why do reefs bleach"}
    )
    assert response.status_code == 502

    listing = await client.get(TRACES, params={"project_id": str(corpus)})
    traces = listing.json()
    assert traces, "the failed query must have left a trace"

    trace = await _trace(client, traces[0]["id"])
    assert trace["status"] == "error"
    assert trace["error_code"] == "MODEL_UNAVAILABLE"

    by_stage = {span["stage"]: span for span in trace["spans"]}
    # Retrieval succeeded before generation blew up; both facts are recorded.
    assert by_stage["retrieval"]["status"] == "ok"
    assert by_stage["generation"]["status"] == "error"
    assert by_stage["generation"]["error_code"] == "MODEL_UNAVAILABLE"


async def test_an_abstention_is_traced_as_a_success(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    """Abstaining is a correct outcome, not a failure."""
    answer = await _ask(client, corpus, query="zzzqqq nonexistent", strategy="keyword")

    trace = await _trace(client, answer["trace_id"])
    assert answer["abstained"] is True
    assert trace["status"] == "ok"
    assert trace["abstained"] is True
    assert trace["metrics"]["evidence_count"] == 0


# --- access and listing ---------------------------------------------------


async def test_traces_are_listed_newest_first(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: UUID
) -> None:
    await _ask(client, corpus, query="first question")
    await _ask(client, corpus, query="second question")

    listing = await client.get(TRACES, params={"project_id": str(corpus)})

    assert listing.status_code == 200
    questions = [item["query_text"] for item in listing.json()]
    assert questions[:2] == ["second question", "first question"]


async def test_an_unknown_trace_is_not_found(client: AsyncClient, app_with_fakes: FastAPI) -> None:
    response = await client.get(f"{TRACES}/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


async def test_a_malformed_trace_id_is_a_validation_error(
    client: AsyncClient, app_with_fakes: FastAPI
) -> None:
    response = await client.get(f"{TRACES}/not-a-uuid")

    assert response.status_code == 422
