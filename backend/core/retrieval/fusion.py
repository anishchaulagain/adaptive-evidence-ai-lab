"""Result fusion for hybrid retrieval (spec section 15C).

Reciprocal Rank Fusion is the default: rank-based, score-scale independent and
deterministic.
"""

from __future__ import annotations

from core.types import RetrievedChunk

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], *, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists, preserving which retriever contributed each hit."""
    raise NotImplementedError


def weighted_score_fusion(
    rankings: list[list[RetrievedChunk]], weights: list[float]
) -> list[RetrievedChunk]:
    """Alpha-weighted fusion over normalised scores."""
    raise NotImplementedError
