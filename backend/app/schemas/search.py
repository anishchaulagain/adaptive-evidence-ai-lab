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
    """Which retriever runs."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    # Hybrid with query-dependent weights and pool size (spec section 16).
    ADAPTIVE = "adaptive"


class QueryAnalysisRead(APIModel):
    """Why the adaptive strategy retrieved the way it did (spec section 14).

    `signals` names the rules that fired, so the choice is explainable rather
    than merely reported.
    """

    query_type: str
    difficulty: float
    ambiguity: float
    requires_exact_match: bool
    requires_multihop: bool
    requires_numeric_reasoning: bool
    entities: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    semantic_weight: float
    keyword_weight: float
    fetch_multiplier: int
    reason: str


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

    # --- fusion provenance (spec section 17) -----------------------------
    # Populated for hybrid retrieval. `retrieval_source` names every arm that
    # found this chunk, and the per-arm rank and score say what each one
    # thought of it, so a fused ranking can be explained rather than trusted.
    retrieval_source: list[str] = Field(default_factory=list)
    fusion_score: float | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None


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
    # Present only for the adaptive strategy, which is the one that makes a
    # choice worth explaining.
    analysis: QueryAnalysisRead | None = None
    results: list[EvidenceItem]
