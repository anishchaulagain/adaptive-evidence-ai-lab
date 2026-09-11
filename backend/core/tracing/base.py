"""Trace construction (spec section 23).

Every pipeline stage opens a span. A stage that is not traced is not
observable, and therefore not finished.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Protocol


class TraceStage(StrEnum):
    QUERY_ANALYSIS = "query_analysis"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"
    KEYWORD_RETRIEVAL = "keyword_retrieval"
    FUSION = "fusion"
    RERANKING = "reranking"
    EVIDENCE_ASSESSMENT = "evidence_assessment"
    MODEL_ROUTING = "model_routing"
    GENERATION = "generation"
    VERIFICATION = "verification"


class TraceRecorder(Protocol):
    """Collects spans for a single query execution.

    `span()` is an async context manager: it records start/end timestamps and
    attaches an error code if the stage raises.
    """

    def span(self, stage: TraceStage, **attributes: Any) -> AsyncIterator[None]: ...

    def record_usage(self, *, input_tokens: int, output_tokens: int, cost: float) -> None: ...
