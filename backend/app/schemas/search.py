"""Retrieval schemas (spec section 17).

Every hit reports its score and which retriever produced it. That is what makes
the strategy comparisons in Phases 4-5 possible, so it is part of the contract
from the first retriever rather than bolted on later.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class SearchStrategy(StrEnum):
    """Which retriever runs. `HYBRID` arrives in Phase 5."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class SearchRequest(APIModel):
    """A retrieval request against one project's chunks."""

    project_id: UUID
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=100)
    strategy: SearchStrategy = SearchStrategy.SEMANTIC


class EvidenceItem(APIModel):
    """A retrieved chunk with the provenance needed to verify it.

    `score` ranks chunks *within one query and strategy* — cosine similarity
    in [-1, 1] for semantic, normalised `ts_rank_cd` in [0, 1) for keyword.
    Larger is a closer match. It is not comparable across queries or across
    strategies, so do not average or threshold it globally.
    """

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    retriever: str
    rank: int
    page: int | None
    char_start: int | None
    char_end: int | None
    # Populated for keyword retrieval; empty for semantic, where nothing
    # lexical matched at all. That contrast is the point of comparing them.
    matched_terms: list[str] = Field(default_factory=list)


class SearchResponse(APIModel):
    """Retrieval results, plus what it cost to produce them."""

    query: str
    strategy: SearchStrategy
    top_k: int
    latency_ms: float
    # Only set when an embedding model actually ran.
    embedding_model: str | None = None
    # The lexemes the query reduced to, for keyword retrieval. Explains why a
    # result matched when stemming or stopword removal changed the query.
    query_terms: list[str] = Field(default_factory=list)
    results: list[EvidenceItem]
