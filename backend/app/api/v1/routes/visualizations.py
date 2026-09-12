"""Visualization payloads (spec sections 24, 37, 75).

Render-ready data for the four views §75 prioritises: trace timeline, evidence
graph, retrieval comparison and evaluation heatmap. The computation lives in
`visualization/`, which is free of the ORM; these endpoints only load rows and
hand them over.

Deliberately separate from the resource endpoints. `/traces/{id}` returns the
trace as data; this returns the same trace shaped for drawing, and a client
that wants one should not be forced to pay for the other.
"""

from __future__ import annotations

import time
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import EmbeddingsDep, PrincipalDep, SessionDep, SettingsDep
from app.models.chunk import Chunk
from app.models.document import Document
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from app.schemas.search import SearchRequest, SearchStrategy
from app.services.evaluation_service import EvaluationService, RunStatus
from app.services.project_service import ProjectService
from app.services.trace_service import TraceService
from core.errors import ErrorCode, ProviderError
from core.retrieval.base import RetrievalQuery
from visualization.evaluations.heatmap import RunInput, build_heatmap
from visualization.evidence_graph.builder import (
    ChunkInput,
    ClaimInput,
    build_evidence_graph,
    graph_summary,
)
from visualization.retrieval.comparison import StrategyResult, build_comparison
from visualization.traces.timeline import SpanInput, build_timeline

router = APIRouter(prefix="/visualizations", tags=["visualizations"])

_COMPARISON_STRATEGIES = (
    SearchStrategy.SEMANTIC,
    SearchStrategy.KEYWORD,
    SearchStrategy.HYBRID,
)


@router.get("/traces/{trace_id}/timeline")
async def trace_timeline(
    trace_id: UUID, principal: PrincipalDep, session: SessionDep
) -> dict[str, object]:
    """Execution timeline for one trace (spec section 24)."""
    trace = await TraceService(session).get(trace_id, principal)
    start = trace.started_at

    timeline = build_timeline(
        trace_id=str(trace.id),
        total_ms=trace.total_latency_ms,
        status=trace.status,
        spans=[
            SpanInput(
                span_id=str(span.id),
                parent_span_id=str(span.parent_span_id) if span.parent_span_id else None,
                stage=span.stage,
                status=span.status,
                offset_ms=round((span.started_at - start).total_seconds() * 1000, 3),
                duration_ms=span.duration_ms,
                error_code=span.error_code,
                attributes=span.attributes,
            )
            for span in trace.spans
        ],
    )
    return {
        "trace_id": timeline.trace_id,
        "total_ms": timeline.total_ms,
        "status": timeline.status,
        "query": trace.query_text,
        "bars": [
            {
                "span_id": bar.span_id,
                "stage": bar.stage,
                "status": bar.status,
                "depth": bar.depth,
                "offset_ms": bar.offset_ms,
                "duration_ms": bar.duration_ms,
                "error_code": bar.error_code,
                "attributes": bar.attributes,
            }
            for bar in timeline.bars
        ],
    }


@router.get("/traces/{trace_id}/evidence-graph")
async def trace_evidence_graph(
    trace_id: UUID, principal: PrincipalDep, session: SessionDep
) -> dict[str, object]:
    """Evidence graph for one answered query (spec section 18)."""
    trace = await TraceService(session).get(trace_id, principal)

    rows = (
        await session.execute(
            select(Chunk, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(trace.retrieved_chunk_ids or []))
        )
    ).all()
    # Preserve retrieval order: the graph shows rank, and a set lookup would
    # lose it.
    by_id = {chunk.id: (chunk, title) for chunk, title in rows}

    chunks = []
    for rank, chunk_id in enumerate(trace.retrieved_chunk_ids or []):
        found = by_id.get(chunk_id)
        if found is None:
            # The chunk was deleted since the trace was recorded. Skipping it
            # is honest: the graph shows what can still be resolved.
            continue
        chunk, title = found
        chunks.append(
            ChunkInput(
                chunk_id=str(chunk.id),
                document_id=str(chunk.document_id),
                document_title=title,
                text=chunk.text,
                page=chunk.page,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                rank=rank,
            )
        )

    graph = build_evidence_graph(
        query=trace.query_text,
        answer=trace.answer,
        chunks=chunks,
        claims=[
            ClaimInput(
                text=str(claim.get("text", "")),
                cited_chunk_ids=[str(c) for c in claim.get("evidence", [])],
            )
            for claim in (trace.claims or [])
        ],
        abstained=trace.abstained,
    )
    return {
        "trace_id": str(trace.id),
        "nodes": [
            {
                "id": node.id,
                "type": str(node.type),
                "label": node.label,
                "metadata": node.metadata,
            }
            for node in graph.nodes
        ],
        "edges": [
            {"source": edge.source, "target": edge.target, "kind": str(edge.kind)}
            for edge in graph.edges
        ],
        "summary": graph_summary(graph),
    }


@router.post("/retrieval-comparison")
async def retrieval_comparison(
    payload: SearchRequest,
    principal: PrincipalDep,
    session: SessionDep,
    embeddings: EmbeddingsDep,
) -> dict[str, object]:
    """Run every strategy on one query and compare their rankings.

    `strategy` on the request is ignored: the point is to run all of them.
    """
    project = await ProjectService(session).get(payload.project_id, principal)
    if embeddings is None:
        raise ProviderError(
            "MISTRAL_API_KEY is not set, so the semantic and hybrid arms cannot run.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
        )

    query = RetrievalQuery(text=payload.query, project_id=project.id, top_k=payload.top_k)
    keyword = PostgresFtsRetriever(session)
    dense = PgVectorRetriever(session, embeddings)

    results = []
    for strategy in _COMPARISON_STRATEGIES:
        started = time.perf_counter()
        match strategy:
            case SearchStrategy.KEYWORD:
                hits = await keyword.retrieve(query)
            case SearchStrategy.SEMANTIC:
                hits = await dense.retrieve(query)
            case _:
                hits = await HybridRetriever(dense, keyword).retrieve(query)
        results.append(
            StrategyResult(
                strategy=str(strategy),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                hits=[(str(hit.provenance.chunk_id), hit.text, hit.score) for hit in hits],
            )
        )

    comparison = build_comparison(results)
    comparison["query"] = payload.query
    comparison["top_k"] = payload.top_k
    return comparison


@router.get("/evaluations/heatmap")
async def evaluation_heatmap(
    project_id: UUID,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    metric: str = Query(default="recall@5"),
) -> dict[str, object]:
    """Strategy x evidence-condition matrix for one metric (spec section 37.6).

    Built only from completed runs. A combination never run is reported as
    `null`, never as a zero — spec section 37 is explicit that benchmark
    numbers must not be fabricated.
    """
    runs = await EvaluationService(session, settings).list_runs(project_id, principal)
    completed = [run for run in runs if run.status == RunStatus.COMPLETED]

    heatmap = build_heatmap(
        [
            RunInput(
                run_id=str(run.id),
                strategy=str(run.config.get("strategy", "unknown")),
                by_condition=run.aggregate_metrics.get("by_condition", {}),
            )
            for run in completed
        ],
        metric=metric,
    )
    return {
        "metric": heatmap.metric,
        "rows": heatmap.rows,
        "columns": heatmap.columns,
        "cells": [
            {
                "row": cell.row,
                "column": cell.column,
                "value": cell.value,
                "n": cell.n,
                "run_id": cell.run_id,
            }
            for cell in heatmap.cells
        ],
        "run_count": len(completed),
    }
