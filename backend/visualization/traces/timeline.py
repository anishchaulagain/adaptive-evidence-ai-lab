"""Execution timeline payload (spec section 24)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visualization.types import Timeline, TimelineBar


@dataclass(frozen=True, slots=True)
class SpanInput:
    """A stored span, flattened out of the ORM."""

    span_id: str
    parent_span_id: str | None
    stage: str
    status: str
    offset_ms: float
    duration_ms: float
    error_code: str | None = None
    attributes: dict[str, Any] | None = None


def build_timeline(
    *, trace_id: str, total_ms: float, status: str, spans: list[SpanInput]
) -> Timeline:
    """Turn stored spans into render-ready bars.

    Depth is resolved by walking parent links once here, so a viewer can indent
    nested stages without reconstructing the tree — and cannot get it wrong.
    A parent that is missing (a truncated trace) leaves its child at depth 0
    rather than failing the whole view.
    """
    by_id = {span.span_id: span for span in spans}

    def depth_of(span: SpanInput) -> int:
        depth = 0
        seen: set[str] = {span.span_id}
        parent = span.parent_span_id
        while parent is not None and parent in by_id and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = by_id[parent].parent_span_id
        return depth

    return Timeline(
        trace_id=trace_id,
        total_ms=total_ms,
        status=status,
        bars=[
            TimelineBar(
                span_id=span.span_id,
                stage=span.stage,
                status=span.status,
                depth=depth_of(span),
                offset_ms=span.offset_ms,
                duration_ms=span.duration_ms,
                error_code=span.error_code,
                attributes=span.attributes or {},
            )
            for span in spans
        ],
    )
