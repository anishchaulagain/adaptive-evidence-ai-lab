"""Visualization endpoints against real data (spec sections 24, 37, 75)."""

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

VIS = "/api/v1/visualizations"

CORPUS = "\n\n".join(
    [
        "Coral reefs bleach when ocean temperatures stay elevated for weeks.",
        "The Antarctic ice sheet lost mass at an accelerating rate.",
        "Sovereign green bond issuance surpassed one trillion dollars.",
        "The central bank held interest rates steady, citing core inflation.",
    ]
).encode()


@pytest.fixture
def fake_pipeline() -> BatchedEmbeddingPipeline:
    return BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=4)


@pytest.fixture
def app_with_fakes(app: FastAPI, fake_pipeline: BatchedEmbeddingPipeline) -> FastAPI:
    async def _embeddings() -> AsyncIterator[BatchedEmbeddingPipeline]:
        yield fake_pipeline

    async def _generator() -> AsyncIterator[GroundedAnswerGenerator]:
        yield GroundedAnswerGenerator(
            FakeChatModel(
                {
                    "answer": "Reefs bleach during prolonged ocean heat.",
                    "claims": [
                        {"text": "Heat drives bleaching.", "evidence": [0]},
                        {"text": "An uncited assertion.", "evidence": []},
                    ],
                }
            )
        )

    app.dependency_overrides[get_embeddings] = _embeddings
    app.dependency_overrides[get_generator] = _generator
    return app


@pytest.fixture
async def answered(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    settings: Settings,
    fake_pipeline: BatchedEmbeddingPipeline,
) -> AsyncIterator[tuple[UUID, str]]:
    """A project with one answered query, yielding (project_id, trace_id)."""
    created = await client.post("/api/v1/projects", json={"name": f"Viz {uuid4().hex[:8]}"})
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

    answer = await client.post(
        "/api/v1/query",
        json={"project_id": str(project_id), "query": "why do reefs bleach"},
    )
    assert answer.status_code == 200, answer.text

    yield project_id, answer.json()["trace_id"]

    async with sessionmaker() as session:
        await session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": str(project_id)})
        await session.commit()


# --- trace timeline -------------------------------------------------------


async def test_the_timeline_renders_stages_with_offsets_and_depth(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    _, trace_id = answered

    response = await client.get(f"{VIS}/traces/{trace_id}/timeline")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_ms"] > 0
    assert body["query"] == "why do reefs bleach"

    stages = [bar["stage"] for bar in body["bars"]]
    assert "retrieval" in stages
    assert "generation" in stages
    offsets = [bar["offset_ms"] for bar in body["bars"]]
    assert offsets == sorted(offsets)
    assert all(bar["depth"] >= 0 for bar in body["bars"])


async def test_the_timeline_carries_stage_metadata(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    _, trace_id = answered

    body = (await client.get(f"{VIS}/traces/{trace_id}/timeline")).json()

    by_stage = {bar["stage"]: bar for bar in body["bars"]}
    assert by_stage["retrieval"]["attributes"]["candidate_count"] > 0


# --- evidence graph -------------------------------------------------------


async def test_the_evidence_graph_chains_query_to_answer(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    """Rebuilt from the stored trace alone, which is what makes an answer
    auditable after the fact."""
    _, trace_id = answered

    response = await client.get(f"{VIS}/traces/{trace_id}/evidence-graph")

    assert response.status_code == 200, response.text
    body = response.json()
    types = {node["type"] for node in body["nodes"]}
    assert {"query", "document", "chunk", "claim", "answer"} <= types

    kinds = {edge["kind"] for edge in body["edges"]}
    assert {"retrieved", "contains", "cites", "supports"} <= kinds


async def test_the_graph_marks_uncited_chunks_and_unsupported_claims(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    """The fake model cites one passage and makes one bare assertion."""
    _, trace_id = answered

    body = (await client.get(f"{VIS}/traces/{trace_id}/evidence-graph")).json()

    summary = body["summary"]
    assert summary["unsupported_claims"] == 1
    assert summary["uncited_chunks"] >= 1

    chunks = [n for n in body["nodes"] if n["type"] == "chunk"]
    assert any(node["metadata"]["cited"] for node in chunks)
    assert any(not node["metadata"]["cited"] for node in chunks)


async def test_graph_chunks_carry_provenance(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    """A graph node that cannot be resolved to a span is not evidence."""
    _, trace_id = answered

    body = (await client.get(f"{VIS}/traces/{trace_id}/evidence-graph")).json()

    for node in (n for n in body["nodes"] if n["type"] == "chunk"):
        assert node["metadata"]["chunk_id"]
        assert node["metadata"]["document_id"]
        assert node["metadata"]["rank"] is not None


async def test_an_unknown_trace_has_no_graph(client: AsyncClient, app_with_fakes: FastAPI) -> None:
    response = await client.get(f"{VIS}/traces/{uuid4()}/evidence-graph")

    assert response.status_code == 404


# --- retrieval comparison -------------------------------------------------


async def test_the_comparison_runs_every_strategy(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    project_id, _ = answered

    response = await client.post(
        f"{VIS}/retrieval-comparison",
        json={"project_id": str(project_id), "query": "coral reefs bleach", "top_k": 5},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["strategy"] for item in body["strategies"]] == [
        "semantic",
        "keyword",
        "hybrid",
    ]
    assert body["rows"]
    for row in body["rows"]:
        assert set(row["ranks"]) == {"semantic", "keyword", "hybrid"}
        assert row["found_by"]


async def test_the_comparison_reports_unique_contributions(
    client: AsyncClient, answered: tuple[UUID, str]
) -> None:
    """This is the number that says whether an arm earns its place."""
    project_id, _ = answered

    body = (
        await client.post(
            f"{VIS}/retrieval-comparison",
            json={"project_id": str(project_id), "query": "ocean temperatures", "top_k": 5},
        )
    ).json()

    overlap = body["overlap"]
    assert overlap["total_chunks"] > 0
    assert set(overlap["unique_to"]) == {"semantic", "keyword", "hybrid"}


async def test_the_comparison_needs_a_provider_key(
    client: AsyncClient, app: FastAPI, answered: tuple[UUID, str]
) -> None:
    project_id, _ = answered

    async def _no_embeddings() -> AsyncIterator[None]:
        yield None

    app.dependency_overrides[get_embeddings] = _no_embeddings
    response = await client.post(
        f"{VIS}/retrieval-comparison",
        json={"project_id": str(project_id), "query": "anything"},
    )
    app.dependency_overrides.pop(get_embeddings, None)

    assert response.status_code == 503


# --- evaluation heatmap ---------------------------------------------------


async def test_the_heatmap_is_built_from_completed_runs(
    client: AsyncClient,
    app_with_fakes: FastAPI,
    answered: tuple[UUID, str],
    settings: Settings,
) -> None:
    project_id, _ = answered

    chunks = (
        await client.get(
            "/api/v1/documents",
            params={"project_id": str(project_id)},
        )
    ).json()
    assert chunks

    dataset = await client.post(
        "/api/v1/evaluations/datasets",
        json={
            "project_id": str(project_id),
            "name": f"Heatmap {uuid4().hex[:8]}",
            "items": [
                {"question": "coral reefs bleach", "evidence_condition": "clean"},
                {"question": "antarctic ice sheet", "evidence_condition": "noisy"},
            ],
        },
    )
    dataset_id = dataset.json()["id"]

    from app.services.evaluation_service import EvaluationService

    created = await client.post(
        "/api/v1/evaluations", json={"dataset_id": dataset_id, "strategy": "keyword"}
    )
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await EvaluationService(session, settings).execute(UUID(created.json()["id"]))

    response = await client.get(
        f"{VIS}/evaluations/heatmap",
        params={"project_id": str(project_id), "metric": "recall@5"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metric"] == "recall@5"
    assert body["rows"] == ["keyword"]
    assert set(body["columns"]) == {"clean", "noisy"}
    assert len(body["cells"]) == len(body["rows"]) * len(body["columns"])
    assert body["run_count"] == 1


async def test_a_project_with_no_runs_has_an_empty_heatmap(
    client: AsyncClient, app_with_fakes: FastAPI
) -> None:
    created = await client.post("/api/v1/projects", json={"name": f"Bare {uuid4().hex[:8]}"})

    response = await client.get(
        f"{VIS}/evaluations/heatmap", params={"project_id": created.json()["id"]}
    )

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["rows"] == []
    assert body["cells"] == []
    assert body["run_count"] == 0
