"""Hybrid retrieval slice: dense + keyword -> RRF -> unified ranking.

Uses the deterministic embedder so the dense arm is reproducible; the keyword
arm, the fusion and the database are all real.
"""

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
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from app.services.embedding_service import EmbeddingService
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from core.retrieval.base import RetrievalQuery
from core.types import RetrieverKind
from tests.fakes import FakeEmbeddingModel

pytestmark = pytest.mark.integration

SEARCH = "/api/v1/search"

CORPUS = "\n\n".join(
    [
        "Deployment halted with error code ERR-5521 during the rollout phase.",
        "Deployment halted with error code ERR-9310 during the rollback phase.",
        "The ingestion service requires version 4.2.1 or later to start.",
        "The NASA GISTEMP dataset records global surface temperature anomalies.",
        "The NOAA MLOST dataset records global surface temperature anomalies.",
        "Warming should be limited to 1.5 degrees above preindustrial levels.",
        "Coral reefs bleach when ocean temperatures stay elevated for weeks.",
        "Sovereign green bond issuance surpassed one trillion dollars.",
    ]
).encode()


@pytest.fixture
def fake_pipeline() -> BatchedEmbeddingPipeline:
    return BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=4)


@pytest.fixture
def app_with_fake_embeddings(app: FastAPI, fake_pipeline: BatchedEmbeddingPipeline) -> FastAPI:
    async def _override() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    app.dependency_overrides[get_embeddings] = _override
    return app


@pytest.fixture
async def corpus(
    client: AsyncClient, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> AsyncIterator[UUID]:
    """A project ingested, chunked and embedded."""
    created = await client.post("/api/v1/projects", json={"name": f"Hybrid {uuid4().hex[:8]}"})
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
    client: AsyncClient, project_id: UUID, query: str, strategy: str = "hybrid", **kwargs: object
) -> dict[str, Any]:
    response = await client.post(
        SEARCH,
        json={
            "project_id": str(project_id),
            "query": query,
            "strategy": strategy,
            **kwargs,
        },
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# --- unified ranking ------------------------------------------------------


async def test_hybrid_returns_a_single_fused_ranking(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    body = await _search(client, corpus, "GISTEMP dataset anomalies")

    assert body["strategy"] == "hybrid"
    results = body["results"]
    assert results
    assert [item["rank"] for item in results] == list(range(len(results)))
    assert results == sorted(results, key=lambda item: -item["fusion_score"])
    assert all(item["retriever"] == "hybrid" for item in results)


async def test_hybrid_reports_both_the_model_and_the_query_terms(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """Both arms ran, so the response must account for both."""
    body = await _search(client, corpus, "GISTEMP dataset")

    assert body["embedding_model"] == "fake-embed"
    assert body["query_terms"]


async def test_top_k_limits_the_fused_result_count(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """Each arm fetches deeper than top_k, so truncation must happen after
    fusion, not before it."""
    body = await _search(client, corpus, "dataset", top_k=3)

    assert len(body["results"]) <= 3


# --- fusion provenance (spec section 17) ----------------------------------


async def test_results_report_which_arms_found_them(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    body = await _search(client, corpus, "GISTEMP dataset anomalies")

    sources = {source for item in body["results"] for source in item["retrieval_source"]}
    assert sources <= {"semantic", "keyword"}
    assert "semantic" in sources

    for item in body["results"]:
        assert item["retrieval_source"], "every fused hit must name its origin"
        assert item["fusion_score"] is not None
        # A per-arm score exists exactly when that arm is listed as a source.
        assert (item["semantic_score"] is not None) == ("semantic" in item["retrieval_source"])
        assert (item["keyword_score"] is not None) == ("keyword" in item["retrieval_source"])


async def test_a_chunk_found_by_both_arms_records_both_ranks(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """The pre-fusion ranks are what make a fused ranking explainable."""
    body = await _search(client, corpus, "GISTEMP dataset anomalies")

    agreed = [item for item in body["results"] if len(item["retrieval_source"]) == 2]
    assert agreed, "expected at least one chunk found by both arms"
    for item in agreed:
        assert item["semantic_rank"] is not None
        assert item["keyword_rank"] is not None


async def test_agreement_between_arms_wins_the_top_rank(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """A chunk both arms found should outrank one only a single arm did — the
    property hybrid retrieval exists to provide."""
    body = await _search(client, corpus, "GISTEMP dataset anomalies")

    top = body["results"][0]
    assert len(top["retrieval_source"]) == 2


async def test_matched_terms_survive_fusion(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    body = await _search(client, corpus, "GISTEMP")

    lexical = [item for item in body["results"] if "keyword" in item["retrieval_source"]]
    assert lexical
    assert all(item["matched_terms"] for item in lexical)


# --- degradation ----------------------------------------------------------


async def test_hybrid_falls_back_to_dense_when_keyword_matches_nothing(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """The keyword arm is conjunctive, so it contributes nothing to most
    natural-language questions. Hybrid must still return dense results."""
    keyword_only = await _search(
        client, corpus, "how do warming seas damage reef ecosystems", strategy="keyword"
    )
    assert keyword_only["results"] == [], "expected the keyword arm to find nothing"

    body = await _search(client, corpus, "how do warming seas damage reef ecosystems")

    assert body["results"], "hybrid must not be emptied by an empty keyword arm"
    assert all(item["retrieval_source"] == ["semantic"] for item in body["results"])


async def test_hybrid_requires_a_provider_key(
    client: AsyncClient, app: FastAPI, corpus: UUID
) -> None:
    """Without the embedding override the real provider resolves; with no key
    configured, hybrid must fail with an attributable code rather than silently
    degrading to keyword-only results."""

    async def _no_embeddings() -> AsyncIterator[None]:
        yield None

    app.dependency_overrides[get_embeddings] = _no_embeddings
    response = await client.post(
        SEARCH,
        json={"project_id": str(corpus), "query": "anything", "strategy": "hybrid"},
    )
    app.dependency_overrides.pop(get_embeddings, None)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "PROVIDER_NOT_CONFIGURED"


async def test_keyword_only_search_still_needs_no_key(
    client: AsyncClient, app: FastAPI, corpus: UUID
) -> None:
    async def _no_embeddings() -> AsyncIterator[None]:
        yield None

    app.dependency_overrides[get_embeddings] = _no_embeddings
    body = await _search(client, corpus, "GISTEMP", strategy="keyword")
    app.dependency_overrides.pop(get_embeddings, None)

    assert body["results"]


# --- retriever composition ------------------------------------------------


async def test_hybrid_retriever_fetches_deeper_than_top_k(
    corpus: UUID, fake_pipeline: BatchedEmbeddingPipeline
) -> None:
    """Fusion can only rescue a chunk that ranked poorly in one arm if that arm
    was asked for more candidates than the caller wants back."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        dense = PgVectorRetriever(session, fake_pipeline)
        deep = await dense.retrieve(RetrievalQuery(text="dataset", project_id=corpus, top_k=20))

        hybrid = HybridRetriever(dense, PostgresFtsRetriever(session))
        fused = await hybrid.retrieve(RetrievalQuery(text="dataset", project_id=corpus, top_k=2))

    assert len(fused) == 2
    # The corpus has more chunks than top_k, so a deeper fetch really happened.
    assert len(deep) > 2
    assert all(hit.retriever is RetrieverKind.HYBRID for hit in fused)
