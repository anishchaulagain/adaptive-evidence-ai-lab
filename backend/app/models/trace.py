"""Execution traces — every AI operation is inspectable (spec sections 23, 48).

A trace holds the query that produced it directly rather than pointing at a
separate `queries` table. The spec lists both, but a query with no trace is not
a thing this platform creates, and splitting them now would mean a join for
every read and two rows to keep consistent for no gain. A `queries` table can
be introduced later if query history needs to outlive trace retention.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    ARRAY,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Trace(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """One end-to-end query execution."""

    __tablename__ = "traces"

    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Null when no pricing is configured, rather than a fabricated zero that
    # would silently understate the cost of a benchmark run.
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    model: Mapped[str | None] = mapped_column(String(150), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(150), nullable=True)

    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    abstained: Mapped[bool] = mapped_column(nullable=False, default=False)

    # The answer and what it rested on. Stored because spec section 18 requires
    # every answer to be traceable to its evidence, and a trace that records
    # only timings cannot reconstruct that chain after the fact.
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    claims: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    retrieved_chunk_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(Uuid), nullable=False, default=list
    )
    cited_chunk_ids: Mapped[list[UUID]] = mapped_column(ARRAY(Uuid), nullable=False, default=list)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    attributes: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    spans: Mapped[list[TraceSpan]] = relationship(
        back_populates="trace",
        cascade="all, delete-orphan",
        order_by="TraceSpan.started_at",
        lazy="selectin",
    )


class TraceSpan(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One pipeline stage within a trace.

    `parent_span_id` is self-referential, so nested stages (a retrieval arm
    inside hybrid retrieval) reconstruct as a tree.
    """

    __tablename__ = "trace_spans"

    trace_id: Mapped[UUID] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"), index=True, nullable=False
    )
    parent_span_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("trace_spans.id", ondelete="CASCADE"), nullable=True
    )

    stage: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)

    attributes: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    trace: Mapped[Trace] = relationship(back_populates="spans")
