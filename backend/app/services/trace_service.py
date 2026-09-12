"""Trace persistence and retrieval (spec section 23)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.security import Principal
from app.models.trace import Trace, TraceSpan
from app.services.base import Service
from core.tracing.recorder import TraceRecord

logger = get_logger(__name__)


class TraceService(Service):
    """Stores recorded traces and reads them back for the trace viewer."""

    async def save(
        self,
        record: TraceRecord,
        *,
        project_id: UUID,
        principal: Principal,
        query_text: str,
        strategy: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        embedding_model: str | None = None,
        evidence_count: int = 0,
        abstained: bool = False,
        answer: str | None = None,
        claims: list[dict[str, object]] | None = None,
        retrieved_chunk_ids: list[UUID] | None = None,
        cited_chunk_ids: list[UUID] | None = None,
    ) -> Trace:
        """Persist a trace and its spans."""
        trace = Trace(
            id=record.trace_id,
            query_text=query_text,
            strategy=strategy,
            status=str(record.status),
            started_at=record.started_at,
            finished_at=record.finished_at,
            total_latency_ms=record.total_latency_ms,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cost=record.cost,
            model=model,
            provider=provider,
            embedding_model=embedding_model,
            evidence_count=evidence_count,
            abstained=abstained,
            answer=answer,
            claims=claims or [],
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            cited_chunk_ids=cited_chunk_ids or [],
            error_code=record.error_code,
            error_message=record.error_message,
            attributes=record.attributes,
            project_id=project_id,
            organization_id=principal.organization_id,
            user_id=principal.user_id,
        )
        self.session.add(trace)
        self.session.add_all(
            [
                TraceSpan(
                    id=span.span_id,
                    trace_id=record.trace_id,
                    parent_span_id=span.parent_span_id,
                    stage=str(span.stage),
                    status=str(span.status),
                    started_at=span.started_at,
                    finished_at=span.finished_at,
                    duration_ms=span.duration_ms,
                    attributes=span.attributes,
                    error_code=span.error_code,
                    error_message=span.error_message,
                )
                for span in record.spans
            ]
        )
        await self.session.commit()

        logger.info(
            "trace.saved",
            trace_id=str(record.trace_id),
            status=str(record.status),
            spans=len(record.spans),
            total_latency_ms=record.total_latency_ms,
        )
        return trace

    async def get(self, trace_id: UUID, principal: Principal) -> Trace:
        """Fetch a trace the principal owns, with its spans.

        Scoped to the organization for the same reason projects are: a trace
        contains the query text and the evidence that answered it.
        """
        trace = await self.session.scalar(
            select(Trace).where(
                Trace.id == trace_id,
                Trace.organization_id == principal.organization_id,
            )
        )
        if trace is None:
            raise NotFoundError("Trace not found.", details={"trace_id": str(trace_id)})
        return trace

    async def list_for_project(
        self, project_id: UUID, principal: Principal, *, limit: int = 50
    ) -> list[Trace]:
        result = await self.session.scalars(
            select(Trace)
            .where(
                Trace.project_id == project_id,
                Trace.organization_id == principal.organization_id,
            )
            .order_by(Trace.started_at.desc())
            .limit(limit)
        )
        return list(result)
