"""Trace construction (spec sections 23, 48).

Every pipeline stage opens a span. A stage that is not traced is not
observable, and therefore not finished.

Provider- and database-free: the recorder collects spans in memory during a
query, and persistence is the application layer's job. That keeps tracing
usable from workers, scripts and experiment runners, not just from a request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Protocol


class TraceStage(StrEnum):
    """The pipeline stages a trace can contain (spec section 73)."""

    QUERY_ANALYSIS = "query_analysis"
    QUERY_EMBEDDING = "query_embedding"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"
    KEYWORD_RETRIEVAL = "keyword_retrieval"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    RERANKING = "reranking"
    EVIDENCE_ASSESSMENT = "evidence_assessment"
    MODEL_ROUTING = "model_routing"
    GENERATION = "generation"
    VERIFICATION = "verification"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class SpanHandle(Protocol):
    """A span in progress.

    `set()` exists because the facts worth recording — how many candidates a
    retriever returned, which model answered — are only known once the stage
    has run.
    """

    def set(self, **attributes: Any) -> None: ...


class TraceRecorder(Protocol):
    """Collects spans for a single query execution."""

    def span(self, stage: TraceStage, **attributes: Any) -> AsyncIterator[SpanHandle]: ...

    def record_usage(
        self, *, input_tokens: int = 0, output_tokens: int = 0, cost: float | None = None
    ) -> None: ...
