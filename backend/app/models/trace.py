"""Execution traces — every AI operation is traceable (spec sections 23, 48)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Trace(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: query_id, status, started_at, finished_at, total_latency_ms,
    total_input_tokens, total_output_tokens, total_cost, error_code."""

    __tablename__ = "traces"


class TraceSpan(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One pipeline stage within a trace.

    Columns: trace_id, parent_span_id, stage, status, started_at, finished_at,
    latency_ms, input_payload, output_payload, tokens, cost, error_code.
    """

    __tablename__ = "trace_spans"
