"""Trace schemas (spec sections 23, 24)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class TraceSpanRead(APIModel):
    """Fields: id, parent_span_id, stage, status, started_at, finished_at,
    latency_ms, tokens, cost, error_code, input_payload, output_payload."""


class TraceRead(IdentifiedModel):
    """Fields: query_id, status, spans, total_latency_ms, total_tokens,
    total_cost, error_code."""
