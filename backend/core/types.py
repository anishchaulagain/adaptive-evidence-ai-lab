"""Shared pipeline value objects.

Plain dataclasses, deliberately free of SQLAlchemy and FastAPI so the pipeline
stays usable from scripts, notebooks and experiment runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class RetrieverKind(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a piece of text came from. Required on every retrieved item."""

    document_id: UUID
    chunk_id: UUID
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section: str | None = None


@dataclass(frozen=True, slots=True)
class RetrieverContribution:
    """What one retriever contributed to a fused result (spec section 17).

    Keeping the pre-fusion rank and score per retriever is what lets the
    platform answer "did semantic retrieval miss this, and did keyword recover
    it?" — the question fusion exists to be judged on.
    """

    retriever: RetrieverKind
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieval hit, carrying enough detail to explain *why* it was chosen
    (spec section 17)."""

    text: str
    score: float
    retriever: RetrieverKind
    provenance: Provenance
    rank: int
    rerank_score: float | None = None
    # Which query terms actually matched. Empty for semantic retrieval, where
    # nothing lexical matched at all — that distinction is the point.
    matched_terms: tuple[str, ...] = ()
    # Set by fusion only. `contributions` records every retriever that found
    # this chunk, with the rank and score it gave, so a fused ranking stays
    # explainable rather than becoming an opaque number.
    fusion_score: float | None = None
    contributions: tuple[RetrieverContribution, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def contribution(self, retriever: RetrieverKind) -> RetrieverContribution | None:
        """The named retriever's contribution, or None if it did not find this."""
        for item in self.contributions:
            if item.retriever is retriever:
                return item
        return None


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Output of the adaptive query analyzer (spec section 14).

    Fields: intent, difficulty, requires_recency, entities, suggested strategy.
    """

    intent: str
    difficulty: float
    suggested_strategy: RetrieverKind
    entities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
