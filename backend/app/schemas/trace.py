"""Trace schemas (spec sections 23, 24, 25).

Shaped for the execution timeline the UI draws: every span carries an absolute
start, a duration, and an offset from the trace start, so a caller can render
the timeline without recomputing it from timestamps.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel, IdentifiedModel


class TraceSpanRead(APIModel):
    """One pipeline stage."""

    id: UUID
    parent_span_id: UUID | None
    stage: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    # Milliseconds from the start of the trace — the timeline's x position.
    offset_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


class TraceMetrics(APIModel):
    """The per-stage figures spec section 25 requires."""

    total_latency_ms: float
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    reranking_latency_ms: float | None = None
    verification_latency_ms: float | None = None

    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Null unless token pricing is configured; never a fabricated zero.
    cost: float | None = None

    evidence_count: int


class TraceRead(IdentifiedModel):
    """A full execution trace."""

    query_text: str
    strategy: str | None
    status: str
    abstained: bool

    started_at: datetime
    finished_at: datetime

    model: str | None
    provider: str | None
    embedding_model: str | None

    metrics: TraceMetrics
    spans: list[TraceSpanRead] = Field(default_factory=list)

    error_code: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceSummary(IdentifiedModel):
    """A trace without its spans, for listings."""

    query_text: str
    strategy: str | None
    status: str
    abstained: bool
    started_at: datetime
    total_latency_ms: float
    total_tokens: int
    cost: float | None
    evidence_count: int
    error_code: str | None = None
