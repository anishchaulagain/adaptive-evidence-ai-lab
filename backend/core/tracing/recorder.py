"""In-memory trace recording.

Collects one span per pipeline stage, then hands back a `TraceRecord` for the
application layer to persist. Spans nest: a stage opened inside another records
its parent, so the tree in spec section 23 is reconstructable and the timeline
in section 24 can be drawn from real timings.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from core.errors import DomainError
from core.tracing.base import SpanStatus, TraceStage


@dataclass(slots=True)
class SpanRecord:
    """One completed pipeline stage."""

    # Order in which the stage *started*. A wall-clock timestamp is not enough:
    # on a coarse system clock two stages can share a timestamp, and sorting by
    # it then silently falls back to completion order — which puts a nested
    # stage before the parent that contains it.
    sequence: int
    span_id: UUID
    parent_span_id: UUID | None
    stage: TraceStage
    status: SpanStatus
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class TraceRecord:
    """A complete execution trace, ready to persist."""

    trace_id: UUID
    status: SpanStatus
    started_at: datetime
    finished_at: datetime
    total_latency_ms: float
    spans: list[SpanRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # None when no pricing is configured. An invented number would be worse
    # than no number in a platform that reports cost as a result.
    cost: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def stage(self, stage: TraceStage) -> SpanRecord | None:
        """The first span for a stage, or None if it never ran."""
        for span in self.spans:
            if span.stage is stage:
                return span
        return None

    def latency_of(self, stage: TraceStage) -> float | None:
        span = self.stage(stage)
        return span.duration_ms if span else None


class _Span:
    """Mutable handle for a span that is still open."""

    __slots__ = ("attributes",)

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)


class TraceCollector:
    """Implements `TraceRecorder`, accumulating spans in memory."""

    def __init__(self, trace_id: UUID | None = None) -> None:
        self.trace_id = trace_id or uuid4()
        self._spans: list[SpanRecord] = []
        self._stack: list[UUID] = []
        self._sequence = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost: float | None = None
        self._started_at = datetime.now(UTC)
        self._started_perf = time.perf_counter()
        self._attributes: dict[str, Any] = {}

    def set(self, **attributes: Any) -> None:
        """Attach trace-level facts, such as the strategy or the model used."""
        self._attributes.update(attributes)

    def record_usage(
        self, *, input_tokens: int = 0, output_tokens: int = 0, cost: float | None = None
    ) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if cost is not None:
            self._cost = (self._cost or 0.0) + cost

    @asynccontextmanager
    async def span(self, stage: TraceStage, **attributes: Any) -> AsyncIterator[_Span]:
        """Record a stage, including how it failed.

        A raising stage still produces a span, with its error code attached and
        the exception re-raised: spec section 47 requires failures to appear in
        traces, and an untraced failure is the one hardest to diagnose.
        """
        span_id = uuid4()
        parent = self._stack[-1] if self._stack else None
        sequence = self._sequence
        self._sequence += 1
        handle = _Span()
        handle.attributes.update(attributes)

        started_at = datetime.now(UTC)
        started = time.perf_counter()
        self._stack.append(span_id)
        status = SpanStatus.OK
        error_code: str | None = None
        error_message: str | None = None
        try:
            yield handle
        except DomainError as exc:
            status = SpanStatus.ERROR
            error_code = str(exc.code)
            error_message = exc.message
            raise
        except Exception as exc:
            status = SpanStatus.ERROR
            error_code = "INTERNAL_ERROR"
            error_message = str(exc)
            raise
        finally:
            self._stack.pop()
            self._spans.append(
                SpanRecord(
                    sequence=sequence,
                    span_id=span_id,
                    parent_span_id=parent,
                    stage=stage,
                    status=status,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    attributes=dict(handle.attributes),
                    error_code=error_code,
                    error_message=error_message,
                )
            )

    def finish(
        self, *, error_code: str | None = None, error_message: str | None = None
    ) -> TraceRecord:
        """Close the trace and return it for persistence.

        The trace fails if it was told it failed, or if any span did — a stage
        error that never surfaced at the top level is still a failed execution.
        """
        failed = error_code is not None or any(
            span.status is SpanStatus.ERROR for span in self._spans
        )
        return TraceRecord(
            trace_id=self.trace_id,
            status=SpanStatus.ERROR if failed else SpanStatus.OK,
            started_at=self._started_at,
            finished_at=datetime.now(UTC),
            total_latency_ms=round((time.perf_counter() - self._started_perf) * 1000, 3),
            # Ordered by start sequence so the timeline reads in execution
            # order rather than completion order, which nesting would otherwise
            # jumble whenever the clock is too coarse to separate them.
            spans=sorted(self._spans, key=lambda span: span.sequence),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost=self._cost,
            error_code=error_code,
            error_message=error_message,
            attributes=dict(self._attributes),
        )
