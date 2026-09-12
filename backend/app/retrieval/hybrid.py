"""Hybrid retrieval (spec sections 15C, 70).

Runs the dense and keyword arms over the same query and fuses their rankings
with RRF. Composes two `Retriever`s rather than reimplementing either, so the
arms stay independently testable and independently replaceable — a true BM25
backend or Qdrant would drop in without touching this module.
"""

from __future__ import annotations

from core.retrieval.base import RetrievalQuery, Retriever
from core.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from core.types import RetrievedChunk

# Each arm retrieves deeper than the final top_k, or fusion would only ever see
# candidates that already ranked highly — and a chunk the dense arm placed
# 12th, which keyword ranked 1st, is exactly the case hybrid exists to catch.
DEFAULT_FETCH_MULTIPLIER = 3
MIN_FETCH_K = 20


class HybridRetriever:
    """Implements `Retriever` by fusing a dense and a keyword retriever."""

    name = "hybrid_rrf"

    def __init__(
        self,
        dense: Retriever,
        keyword: Retriever,
        *,
        k: int = RRF_K,
        fetch_multiplier: int = DEFAULT_FETCH_MULTIPLIER,
        weights: list[float] | None = None,
    ) -> None:
        self._dense = dense
        self._keyword = keyword
        self._k = k
        self._fetch_multiplier = fetch_multiplier
        self._weights = weights

    def _fetch_k(self, top_k: int) -> int:
        return max(top_k * self._fetch_multiplier, MIN_FETCH_K)

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Retrieve from both arms, fuse, and return the best `top_k`.

        The arms run sequentially rather than concurrently: they share one
        `AsyncSession`, which does not support concurrent operations, and the
        embedding round trip dominates the total anyway — overlapping a ~10ms
        keyword query with a ~400ms embedding call would not repay running two
        sessions and the connection accounting that needs.
        """
        deep = RetrievalQuery(
            text=query.text,
            project_id=query.project_id,
            top_k=self._fetch_k(query.top_k),
            filters=query.filters,
        )

        dense_hits = await self._dense.retrieve(deep)
        # The keyword arm is conjunctive, so an empty result is routine rather
        # than a failure; fusion then simply reproduces the dense ranking.
        keyword_hits = await self._keyword.retrieve(deep)

        fused = reciprocal_rank_fusion([dense_hits, keyword_hits], k=self._k, weights=self._weights)
        return fused[: query.top_k]
