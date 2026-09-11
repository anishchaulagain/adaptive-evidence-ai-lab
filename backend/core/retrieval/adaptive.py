"""Adaptive hybrid retrieval (spec section 16).

Chooses strategy, top_k and fusion weights from the query analysis and from
evidence quality, escalating only when the cheap path proves insufficient.
"""

from __future__ import annotations

from typing import Protocol

from core.retrieval.base import RetrievalQuery
from core.types import QueryAnalysis, RetrievedChunk


class AdaptiveRetrievalPolicy(Protocol):
    """Decides the retrieval plan, and whether another round is warranted."""

    def plan(self, analysis: QueryAnalysis, query: RetrievalQuery) -> RetrievalQuery: ...

    def should_escalate(self, results: list[RetrievedChunk]) -> bool: ...
