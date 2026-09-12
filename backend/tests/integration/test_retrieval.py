"""Dense retrieval slice: ingest -> embed -> pgvector search.

Runs against real Postgres with a real HNSW index, using a deterministic
embedder instead of the provider — so the vector column, the index and the
distance-to-score conversion are all genuinely exercised without a key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select, text

from app.api.deps import get_embeddings
from app.core.config import Settings
from app.db.session import get_sessionmaker
from app.models.chunk import EMBEDDING_DIMENSIONS, Chunk
from app.services.embedding_service import EmbeddingService
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from core.errors import ErrorCode, ProviderError
from tests.fakes import FakeEmbeddingModel

pytestmark = pytest.mark.integration

SEARCH = "/api/v1/search"

CORPUS = {
    "climate.txt": (
        b"Global surface temperature has increased above preindustrial levels.\n\n"
        b"Emissions must fall sharply by 2030 to limit warming."
    ),
    "finance.txt": (
        b"Adaptation finance remains below assessed developing country needs.\n\n"
        b"Investment in renewable capacity reached a record last year."
    ),
}


@pytest.fixture
def fake_pipeline() -> BatchedEmbeddingPipeline:
    return BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=4)


@pytest.fixture
def app_with_fake_embeddings(app: FastAPI, fake_pipeline: BatchedEmbeddingPipeline) -> FastAPI:
    """Substitute the embedder so no provider key or network is needed."""

    async def _override() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    app.dependency_overrides[get_embeddings] = _override
    return app


@pytest.fixture
async def corpus(
    client: AsyncClient, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> AsyncIterator[UUID]:
    """A project with two ingested, embedded documents."""
    created = await client.post("/api/v1/projects", json={"name": f"Retrieval {uuid4().hex[:8]}"})
    project_id = UUID(created.json()["id"])

    from app.workers.tasks.ingestion import ingest_document

    sessionmaker = get_sessionmaker()
    for filename, content in CORPUS.items():
        response = await client.post(
            "/api/v1/documents/upload",
            params={"project_id": str(project_id)},
            files={"file": (filename, content, "text/plain")},
        )
        document_id = UUID(response.json()["id"])
        await ingest_document({"settings": settings}, document_id)
        async with sessionmaker() as session:
            await EmbeddingService(session, settings, fake_pipeline).embed_document(document_id)

    yield project_id

    async with sessionmaker() as session:
        await session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": str(project_id)})
        await session.commit()


# --- embedding ------------------------------------------------------------


async def test_embedding_writes_vectors_and_records_the_model(
    corpus: UUID, settings: Settings
) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        chunks = list(await session.scalars(select(Chunk).where(Chunk.project_id == corpus)))

    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSIONS
        # Recording the model is what makes re-embedding after a model change
        # a resumable operation rather than a full rebuild.
        assert chunk.embedding_model == "fake-embed"


async def test_embedding_is_resumable_and_skips_finished_chunks(
    corpus: UUID, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> None:
    """A second run must send nothing to the provider — embedding is the
    billable stage, so re-running a job must not re-pay for it."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        document_id = (
            await session.scalars(
                select(Chunk.document_id).where(Chunk.project_id == corpus).limit(1)
            )
        ).one()

        written = await EmbeddingService(session, settings, fake_pipeline).embed_document(
            document_id
        )

    assert written == 0


async def test_only_missing_chunks_are_re_embedded(
    corpus: UUID, settings: Settings, fake_pipeline: BatchedEmbeddingPipeline
) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        chunk = (
            await session.scalars(
                select(Chunk).where(Chunk.project_id == corpus).order_by(Chunk.ordinal)
            )
        ).first()
        assert chunk is not None
        document_id = chunk.document_id
        chunk.embedding = None
        chunk.embedding_model = None
        await session.commit()

    async with sessionmaker() as session:
        written = await EmbeddingService(session, settings, fake_pipeline).embed_document(
            document_id
        )

    assert written == 1


async def test_missing_provider_key_fails_with_an_attributable_code(
    corpus: UUID, settings: Settings
) -> None:
    """Without a key the failure must name the cause, not surface as a 500."""
    unset = settings.model_copy(update={"MISTRAL_API_KEY": None})
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        document_id = (
            await session.scalars(
                select(Chunk.document_id).where(Chunk.project_id == corpus).limit(1)
            )
        ).one()
        service = EmbeddingService(session, unset)

        with pytest.raises(ProviderError) as excinfo:
            await service.embed_document(document_id)

    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


# --- retrieval ------------------------------------------------------------


async def test_search_returns_scored_ranked_results(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    response = await client.post(
        SEARCH,
        json={"project_id": str(corpus), "query": "renewable capacity investment"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["strategy"] == "semantic"
    assert body["embedding_model"] == "fake-embed"
    assert body["latency_ms"] >= 0

    results = body["results"]
    assert results, "expected at least one hit"
    assert [item["rank"] for item in results] == list(range(len(results)))
    # Scores must be ordered best-first and comparable across retrievers.
    assert results == sorted(results, key=lambda item: -item["score"])
    assert all(item["retriever"] == "semantic" for item in results)


async def test_the_most_similar_chunk_ranks_first(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """Querying a chunk's own text must return that chunk first, at a score of
    essentially 1 — this is what proves the index and distance maths agree."""
    query = "Investment in renewable capacity reached a record last year."

    response = await client.post(SEARCH, json={"project_id": str(corpus), "query": query})

    top = response.json()["results"][0]
    assert top["text"] == query
    assert top["score"] == pytest.approx(1.0, abs=1e-6)


async def test_results_carry_provenance(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """A hit without offsets cannot be verified against the source."""
    response = await client.post(SEARCH, json={"project_id": str(corpus), "query": "emissions"})

    for item in response.json()["results"]:
        assert item["chunk_id"]
        assert item["document_id"]
        assert item["char_start"] is not None
        assert item["char_end"] is not None


async def test_top_k_limits_the_result_count(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    response = await client.post(
        SEARCH,
        json={"project_id": str(corpus), "query": "emissions", "top_k": 2},
    )

    assert len(response.json()["results"]) == 2


async def test_unembedded_chunks_are_not_returned(
    client: AsyncClient,
    app_with_fake_embeddings: FastAPI,
    corpus: UUID,
    settings: Settings,
) -> None:
    """Chunks awaiting embedding must be absent rather than ranked arbitrarily."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        chunks = list(await session.scalars(select(Chunk).where(Chunk.project_id == corpus)))
        for chunk in chunks:
            chunk.embedding = None
        await session.commit()

    response = await client.post(SEARCH, json={"project_id": str(corpus), "query": "emissions"})

    assert response.json()["results"] == []


async def test_search_is_scoped_to_the_project(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, corpus: UUID
) -> None:
    """Spec section 42: a valid project ID from another organization must not
    leak, and neither must another project's evidence."""
    other = await client.post("/api/v1/projects", json={"name": f"Empty {uuid4().hex[:8]}"})
    other_id = other.json()["id"]

    response = await client.post(SEARCH, json={"project_id": other_id, "query": "emissions"})

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_unknown_project_is_not_found(
    client: AsyncClient, app_with_fake_embeddings: FastAPI
) -> None:
    response = await client.post(SEARCH, json={"project_id": str(uuid4()), "query": "emissions"})

    assert response.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "missing project"},
        {"project_id": "not-a-uuid", "query": "bad id"},
        {"project_id": str(uuid4()), "query": ""},
        {"project_id": str(uuid4()), "query": "ok", "top_k": 0},
        {"project_id": str(uuid4()), "query": "ok", "top_k": 101},
    ],
    ids=["no-project", "bad-uuid", "empty-query", "top-k-zero", "top-k-too-large"],
)
async def test_invalid_requests_are_rejected(
    client: AsyncClient, app_with_fake_embeddings: FastAPI, payload: dict[str, object]
) -> None:
    response = await client.post(SEARCH, json=payload)

    assert response.status_code == 422
