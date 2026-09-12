"""Evaluation lab slice: dataset -> run -> metrics (spec sections 26-28, 74).

Uses the deterministic embedder so retrieval is reproducible; the metrics, the
database and the retrieval stack are all real.
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
from app.services.embedding_service import EmbeddingService
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from tests.fakes import FakeEmbeddingModel

pytestmark = pytest.mark.integration

EVALUATIONS = "/api/v1/evaluations"

PASSAGES = [
    "Coral reefs bleach when ocean temperatures stay elevated for weeks.",
    "The Antarctic ice sheet lost mass at an accelerating rate.",
    "Sovereign green bond issuance surpassed one trillion dollars.",
    "The central bank held interest rates steady, citing core inflation.",
    "Photosynthesis converts light energy into chemical energy as glucose.",
]
CORPUS = "\n\n".join(PASSAGES).encode()


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
) -> AsyncIterator[tuple[UUID, list[dict[str, Any]]]]:
    """A project with an embedded corpus, plus its chunks for building gold sets."""
    created = await client.post("/api/v1/projects", json={"name": f"Evaluation {uuid4().hex[:8]}"})
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

    chunks = (
        await client.get(
            f"/api/v1/documents/{document_id}/chunks",
            params={"project_id": str(project_id)},
        )
    ).json()

    yield project_id, chunks

    async with sessionmaker() as session:
        await session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": str(project_id)})
        await session.commit()


def _chunk_id(chunks: list[dict[str, Any]], needle: str) -> str:
    for chunk in chunks:
        if needle.lower() in chunk["text"].lower():
            return str(chunk["id"])
    raise AssertionError(f"no chunk containing {needle!r}")


async def _make_dataset(
    client: AsyncClient, project_id: UUID, items: list[dict[str, Any]]
) -> dict[str, Any]:
    response = await client.post(
        f"{EVALUATIONS}/datasets",
        json={
            "project_id": str(project_id),
            "name": f"Dataset {uuid4().hex[:8]}",
            "items": items,
        },
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def _run(
    client: AsyncClient, dataset_id: str, settings: Settings, **config: Any
) -> dict[str, Any]:
    """Create and execute a run synchronously, as the worker would."""
    created = await client.post(EVALUATIONS, json={"dataset_id": dataset_id, **config})
    assert created.status_code == 201, created.text
    run_id = UUID(created.json()["id"])

    from app.services.evaluation_service import EvaluationService

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await EvaluationService(session, settings).execute(run_id)

    fetched = await client.get(f"{EVALUATIONS}/{run_id}")
    assert fetched.status_code == 200
    body: dict[str, Any] = fetched.json()
    return body


# --- datasets -------------------------------------------------------------


async def test_a_dataset_stores_items_with_gold_and_conditions(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: tuple[UUID, list[dict[str, Any]]]
) -> None:
    project_id, chunks = corpus

    dataset = await _make_dataset(
        client,
        project_id,
        [
            {
                "question": "why do reefs bleach",
                "expected_answer": "prolonged ocean heat",
                "gold_chunks": [_chunk_id(chunks, "Coral reefs")],
                "evidence_condition": "clean",
                "difficulty": "easy",
            },
            {
                "question": "what is the boiling point of mercury",
                "gold_chunks": [],
                "evidence_condition": "incomplete",
            },
        ],
    )

    assert dataset["item_count"] == 2
    assert dataset["items"][0]["evidence_condition"] == "clean"
    assert dataset["items"][0]["gold_chunks"]
    assert dataset["items"][1]["gold_chunks"] == []


async def test_duplicate_dataset_names_conflict(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: tuple[UUID, list[dict[str, Any]]]
) -> None:
    project_id, _ = corpus
    payload = {
        "project_id": str(project_id),
        "name": "Fixed name",
        "items": [{"question": "q"}],
    }
    await client.post(f"{EVALUATIONS}/datasets", json=payload)

    response = await client.post(f"{EVALUATIONS}/datasets", json=payload)

    assert response.status_code == 409


async def test_a_dataset_needs_at_least_one_item(
    client: AsyncClient, app_with_fakes: FastAPI, corpus: tuple[UUID, list[dict[str, Any]]]
) -> None:
    project_id, _ = corpus

    response = await client.post(
        f"{EVALUATIONS}/datasets",
        json={"project_id": str(project_id), "name": "Empty", "items": []},
    )

    assert response.status_code == 422


# --- runs -----------------------------------------------------------------


async def test_a_run_computes_retrieval_metrics(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    project_id, chunks = corpus
    dataset = await _make_dataset(
        client,
        project_id,
        [
            {
                "question": "coral reefs bleach ocean temperatures",
                "gold_chunks": [_chunk_id(chunks, "Coral reefs")],
            },
            {
                "question": "central bank interest rates inflation",
                "gold_chunks": [_chunk_id(chunks, "central bank")],
            },
        ],
    )

    run = await _run(client, dataset["id"], settings, strategy="keyword", top_k=5)

    assert run["status"] == "completed"
    assert run["item_count"] == 2
    assert run["failed_count"] == 0

    overall = run["aggregate_metrics"]["overall"]
    assert overall["recall@5"]["mean"] == pytest.approx(1.0)
    assert overall["mrr"]["mean"] == pytest.approx(1.0)
    assert overall["ndcg@5"]["mean"] == pytest.approx(1.0)
    # Every metric reports how many items defined it.
    assert overall["recall@5"]["n"] == 2


async def test_the_config_is_frozen_onto_the_run(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    """A run's meaning must not change because project defaults were edited
    afterwards (spec principle 2)."""
    project_id, _ = corpus
    dataset = await _make_dataset(client, project_id, [{"question": "anything"}])

    run = await _run(client, dataset["id"], settings, strategy="keyword", top_k=3)

    assert run["config"] == {"strategy": "keyword", "top_k": 3, "generate": False}


async def test_metrics_are_undefined_not_zero_without_gold(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    """An item with no gold has no recall. A fabricated zero would understate
    the system under test."""
    project_id, _ = corpus
    dataset = await _make_dataset(
        client,
        project_id,
        [{"question": "unanswerable question", "evidence_condition": "incomplete"}],
    )

    run = await _run(client, dataset["id"], settings, strategy="keyword")

    overall = run["aggregate_metrics"]["overall"]
    assert overall["recall@5"]["mean"] is None
    assert overall["recall@5"]["n"] == 0


async def test_results_are_reported_per_evidence_condition(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    """Pooling CLEAN with ADVERSARIAL hides the contrast the platform exists
    to measure (spec section 28)."""
    project_id, chunks = corpus
    dataset = await _make_dataset(
        client,
        project_id,
        [
            {
                "question": "coral reefs bleach",
                "gold_chunks": [_chunk_id(chunks, "Coral reefs")],
                "evidence_condition": "clean",
            },
            {
                "question": "zzzqqq nothing matches this",
                "gold_chunks": [_chunk_id(chunks, "Antarctic")],
                "evidence_condition": "adversarial",
            },
        ],
    )

    run = await _run(client, dataset["id"], settings, strategy="keyword", top_k=5)

    by_condition = run["aggregate_metrics"]["by_condition"]
    assert set(by_condition) == {"clean", "adversarial"}
    assert by_condition["clean"]["recall@5"]["mean"] == pytest.approx(1.0)
    assert by_condition["adversarial"]["recall@5"]["mean"] == pytest.approx(0.0)


async def test_per_item_results_record_what_was_retrieved(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    """An aggregate that cannot be drilled into is not diagnosable."""
    project_id, chunks = corpus
    gold = _chunk_id(chunks, "Coral reefs")
    dataset = await _make_dataset(
        client,
        project_id,
        [{"question": "coral reefs bleach", "gold_chunks": [gold]}],
    )

    run = await _run(client, dataset["id"], settings, strategy="keyword", top_k=5)
    results = (await client.get(f"{EVALUATIONS}/{run['id']}/results")).json()

    assert len(results) == 1
    assert gold in results[0]["retrieved_chunk_ids"]
    assert results[0]["metrics"]["recall@5"] == pytest.approx(1.0)
    assert results[0]["failure_category"] is None


async def test_a_retrieval_miss_is_categorised(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    """Failures are counted by cause, so a bad score says *which* stage failed
    (spec section 36)."""
    project_id, chunks = corpus
    dataset = await _make_dataset(
        client,
        project_id,
        [
            {
                "question": "zzzqqq nothing matches",
                "gold_chunks": [_chunk_id(chunks, "Coral reefs")],
            }
        ],
    )

    run = await _run(client, dataset["id"], settings, strategy="keyword")
    results = (await client.get(f"{EVALUATIONS}/{run['id']}/results")).json()

    assert results[0]["failure_category"] == "retrieval_miss"
    assert run["aggregate_metrics"]["overall"]["failure_categories"] == {"retrieval_miss": 1}


async def test_a_run_needs_no_provider_key_for_retrieval_metrics(
    client: AsyncClient,
    app: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    """Keyword retrieval uses no provider, so retrieval quality is measurable
    with no key configured at all."""
    project_id, chunks = corpus
    dataset = await _make_dataset(
        client,
        project_id,
        [
            {
                "question": "coral reefs bleach",
                "gold_chunks": [_chunk_id(chunks, "Coral reefs")],
            }
        ],
    )

    run = await _run(client, dataset["id"], settings, strategy="keyword")

    assert run["status"] == "completed"
    assert run["aggregate_metrics"]["overall"]["recall@5"]["mean"] is not None


# --- access ---------------------------------------------------------------


async def test_runs_are_listed_for_a_project(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    corpus: tuple[UUID, list[dict[str, Any]]],
    settings: Settings,
) -> None:
    project_id, _ = corpus
    dataset = await _make_dataset(client, project_id, [{"question": "q"}])
    await _run(client, dataset["id"], settings, strategy="keyword")

    listing = await client.get(EVALUATIONS, params={"project_id": str(project_id)})

    assert listing.status_code == 200
    assert len(listing.json()) == 1


async def test_an_unknown_run_is_not_found(client: AsyncClient, app_with_fakes: FastAPI) -> None:
    response = await client.get(f"{EVALUATIONS}/{uuid4()}")

    assert response.status_code == 404


async def test_a_run_against_an_unknown_dataset_is_not_found(
    client: AsyncClient, app_with_fakes: FastAPI
) -> None:
    response = await client.post(EVALUATIONS, json={"dataset_id": str(uuid4())})

    assert response.status_code == 404
