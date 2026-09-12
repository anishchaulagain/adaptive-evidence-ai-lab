"""Adaptive retriever (spec sections 16, 76).

Analyses the query, picks a retrieval profile, then fuses the two arms with the
profile's weights. Composes the existing retrievers rather than reimplementing
them, so `adaptive` differs from `hybrid` in exactly one respect — the weights
and pool size come from the query instead of being fixed.

That narrowness is deliberate: it makes the comparison between `hybrid` and
`adaptive` a measurement of the *policy*, not of two different pipelines.
"""

from __future__ import annotations

from core.retrieval.adaptive import RetrievalProfile, RuleBasedRetrievalPolicy
from core.retrieval.base import RetrievalQuery, Retriever
from core.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from core.routing.query_analyzer import RuleBasedQueryAnalyzer
from core.types import QueryAnalysis, RetrievedChunk


class AdaptiveRetriever:
    """Implements `Retriever` with a query-dependent fusion weighting."""

    name = "adaptive_rrf"

    def __init__(
        self,
        dense: Retriever,
        keyword: Retriever,
        *,
        analyzer: RuleBasedQueryAnalyzer | None = None,
        policy: RuleBasedRetrievalPolicy | None = None,
        k: int = RRF_K,
    ) -> None:
        self._dense = dense
        self._keyword = keyword
        self._analyzer = analyzer or RuleBasedQueryAnalyzer()
        self._policy = policy or RuleBasedRetrievalPolicy()
        self._k = k
        # The last decision, so a caller can report why this retrieval ran the
        # way it did without re-analysing the query.
        self.last_analysis: QueryAnalysis | None = None
        self.last_profile: RetrievalProfile | None = None

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        analysis = self._analyzer.analyze(query.text)
        profile = self._policy.profile_for(analysis)
        self.last_analysis = analysis
        self.last_profile = profile

        deep = self._policy.plan(analysis, query)
        dense_hits = await self._dense.retrieve(deep)
        keyword_hits = await self._keyword.retrieve(deep)

        fused = reciprocal_rank_fusion(
            [dense_hits, keyword_hits], k=self._k, weights=profile.weights
        )
        return fused[: query.top_k]
