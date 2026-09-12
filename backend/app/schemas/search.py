"""Retrieval schemas (spec section 17).

Every hit reports its score and which retriever produced it. That is what makes
the strategy comparisons in Phases 4-5 possible, so it is part of the contract
from the first retriever rather than bolted on later.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class SearchRequest(APIModel):
    """A retrieval request against one project's chunks."""

    project_id: UUID
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=100)


class EvidenceItem(APIModel):
    """A retrieved chunk with the provenance needed to verify it.

    `score` is cosine similarity in [-1, 1]; larger is a closer match.
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


class SearchResponse(APIModel):
    """Retrieval results, plus what it cost to produce them."""

    query: str
    strategy: str
    embedding_model: str
    top_k: int
    latency_ms: float
    results: list[EvidenceItem]
