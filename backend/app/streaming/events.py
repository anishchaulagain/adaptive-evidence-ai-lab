"""Typed Server-Sent Event payloads (spec section 45).

The frontend renders pipeline progress from these, so the event names are part
of the API contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StreamEventType(StrEnum):
    QUERY_ANALYZED = "query_analyzed"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"
    KEYWORD_RETRIEVAL = "keyword_retrieval"
    FUSION = "fusion"
    RERANKING = "reranking"
    EVIDENCE_ASSESSED = "evidence_assessed"
    MODEL_SELECTED = "model_selected"
    GENERATING = "generating"
    TOKEN = "token"
    VERIFICATION = "verification"
    DONE = "done"
    ERROR = "error"


class StreamEvent(BaseModel):
    """One SSE frame."""

    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
