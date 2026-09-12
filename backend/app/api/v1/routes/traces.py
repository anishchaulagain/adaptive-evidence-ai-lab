"""Trace retrieval — every AI operation is inspectable (spec sections 23, 44)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from app.api.deps import PrincipalDep, SessionDep
from app.models.trace import Trace
from app.schemas.trace import TraceMetrics, TraceRead, TraceSpanRead, TraceSummary
from app.services.trace_service import TraceService
from core.tracing.base import TraceStage

router = APIRouter(prefix="/traces", tags=["traces"])


def _stage_latency(trace: Trace, stage: TraceStage) -> float | None:
    for span in trace.spans:
        if span.stage == str(stage):
            return span.duration_ms
    return None


def _to_read(trace: Trace) -> TraceRead:
    start = trace.started_at
    return TraceRead(
        id=trace.id,
        created_at=trace.created_at,
        updated_at=trace.updated_at,
        query_text=trace.query_text,
        strategy=trace.strategy,
        status=trace.status,
        abstained=trace.abstained,
        started_at=trace.started_at,
        finished_at=trace.finished_at,
        model=trace.model,
        provider=trace.provider,
        embedding_model=trace.embedding_model,
        metrics=TraceMetrics(
            total_latency_ms=trace.total_latency_ms,
            retrieval_latency_ms=_stage_latency(trace, TraceStage.RETRIEVAL),
            generation_latency_ms=_stage_latency(trace, TraceStage.GENERATION),
            reranking_latency_ms=_stage_latency(trace, TraceStage.RERANKING),
            verification_latency_ms=_stage_latency(trace, TraceStage.VERIFICATION),
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            total_tokens=trace.input_tokens + trace.output_tokens,
            cost=trace.cost,
            evidence_count=trace.evidence_count,
        ),
        spans=[
            TraceSpanRead(
                id=span.id,
                parent_span_id=span.parent_span_id,
                stage=span.stage,
                status=span.status,
                started_at=span.started_at,
                finished_at=span.finished_at,
                duration_ms=span.duration_ms,
                # Precomputed so the client draws the timeline rather than
                # deriving it from timestamps and getting the zero point wrong.
                offset_ms=round((span.started_at - start).total_seconds() * 1000, 3),
                attributes=span.attributes,
                error_code=span.error_code,
                error_message=span.error_message,
            )
            for span in trace.spans
        ],
        error_code=trace.error_code,
        error_message=trace.error_message,
        attributes=trace.attributes,
    )


@router.get("", response_model=list[TraceSummary])
async def list_traces(
    principal: PrincipalDep,
    session: SessionDep,
    project_id: UUID,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[TraceSummary]:
    """Recent traces for a project, newest first."""
    traces = await TraceService(session).list_for_project(project_id, principal, limit=limit)
    return [
        TraceSummary(
            id=trace.id,
            created_at=trace.created_at,
            updated_at=trace.updated_at,
            query_text=trace.query_text,
            strategy=trace.strategy,
            status=trace.status,
            abstained=trace.abstained,
            started_at=trace.started_at,
            total_latency_ms=trace.total_latency_ms,
            total_tokens=trace.input_tokens + trace.output_tokens,
            cost=trace.cost,
            evidence_count=trace.evidence_count,
            error_code=trace.error_code,
        )
        for trace in traces
    ]


@router.get("/{trace_id}", response_model=TraceRead)
async def get_trace(trace_id: UUID, principal: PrincipalDep, session: SessionDep) -> TraceRead:
    """The full execution trace: stages, timings, tokens, cost and errors."""
    trace = await TraceService(session).get(trace_id, principal)
    return _to_read(trace)
