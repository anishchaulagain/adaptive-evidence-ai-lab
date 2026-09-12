"""Adaptive hybrid retrieval (spec sections 16, 76).

Maps a query analysis onto a retrieval profile: how much to weight each arm,
how deep to fetch. Deterministic rules, as §76 directs.

The default weights are **heuristics, not findings**. §76 is explicit that they
are initial values to be made configurable and then tested, so they are
overridable per query type and the profile records which rule produced it.
Whether adapting beats a fixed strategy is a question for the evaluation
harness, not an assumption baked in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.retrieval.base import RetrievalQuery
from core.types import QueryAnalysis, QueryType, RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """How to retrieve for one query."""

    semantic_weight: float
    keyword_weight: float
    # Multiplies the caller's top_k to size each arm's candidate pool. A
    # multi-hop query needs a wider net; a lookup does not.
    fetch_multiplier: int = 3
    reason: str = ""

    def __post_init__(self) -> None:
        if self.semantic_weight < 0 or self.keyword_weight < 0:
            raise ValueError("weights must be non-negative")
        if self.semantic_weight == 0 and self.keyword_weight == 0:
            raise ValueError("at least one arm must carry weight")
        if self.fetch_multiplier < 1:
            raise ValueError("fetch_multiplier must be at least 1")

    @property
    def weights(self) -> list[float]:
        """Fusion weights, semantic first — the order the arms are fused in."""
        return [self.semantic_weight, self.keyword_weight]


# Spec section 16's worked examples, as data rather than as branches. Each is a
# starting hypothesis: conceptual questions lean semantic, exact identifiers
# lean lexical, multi-hop widens the pool.
DEFAULT_PROFILES: dict[QueryType, RetrievalProfile] = {
    QueryType.CONCEPTUAL: RetrievalProfile(0.75, 0.25, 3, "conceptual: semantic-heavy"),
    QueryType.EXACT_ENTITY: RetrievalProfile(0.25, 0.75, 3, "exact entity: keyword-heavy"),
    QueryType.TECHNICAL: RetrievalProfile(0.5, 0.5, 3, "technical: balanced"),
    QueryType.NUMERIC: RetrievalProfile(0.5, 0.5, 3, "numeric: balanced"),
    QueryType.MULTI_HOP: RetrievalProfile(0.5, 0.5, 5, "multi-hop: balanced, wider pool"),
    QueryType.COMPARATIVE: RetrievalProfile(0.6, 0.4, 5, "comparative: wider pool"),
    QueryType.TEMPORAL: RetrievalProfile(0.5, 0.5, 3, "temporal: balanced"),
    # Too little signal to lean either way, so widen rather than guess.
    QueryType.AMBIGUOUS: RetrievalProfile(0.5, 0.5, 5, "ambiguous: balanced, wider pool"),
}

FALLBACK_PROFILE = RetrievalProfile(0.5, 0.5, 3, "default: balanced")


class AdaptiveRetrievalPolicy(Protocol):
    """Decides the retrieval plan, and whether another round is warranted."""

    def profile_for(self, analysis: QueryAnalysis) -> RetrievalProfile: ...

    def plan(self, analysis: QueryAnalysis, query: RetrievalQuery) -> RetrievalQuery: ...

    def should_escalate(self, results: list[RetrievedChunk]) -> bool: ...


@dataclass
class RuleBasedRetrievalPolicy:
    """Implements `AdaptiveRetrievalPolicy` from a profile table."""

    profiles: dict[QueryType, RetrievalProfile] = field(
        default_factory=lambda: dict(DEFAULT_PROFILES)
    )
    fallback: RetrievalProfile = FALLBACK_PROFILE
    # Below this many results, the cheap path plainly did not find enough.
    escalate_below: int = 3

    name = "rule_based"

    def profile_for(self, analysis: QueryAnalysis) -> RetrievalProfile:
        return self.profiles.get(analysis.query_type, self.fallback)

    def plan(self, analysis: QueryAnalysis, query: RetrievalQuery) -> RetrievalQuery:
        """Size the candidate pool for this query.

        Only `top_k` changes: the weights are applied at fusion, not here, so
        a caller still gets back the number of results it asked for.
        """
        profile = self.profile_for(analysis)
        return RetrievalQuery(
            text=query.text,
            project_id=query.project_id,
            top_k=query.top_k * profile.fetch_multiplier,
            filters=query.filters,
        )

    def should_escalate(self, results: list[RetrievedChunk]) -> bool:
        """Whether a second, wider round is worth paying for.

        Escalation is not wired into the retriever yet: whether a second round
        helps is a Phase 14 experiment, and running one by default would spend
        the budget before the question is settled (spec section 20).
        """
        return len(results) < self.escalate_below
