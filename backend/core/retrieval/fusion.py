"""Result fusion for hybrid retrieval (spec section 15C).

Reciprocal Rank Fusion is the default because it needs no score calibration:
the dense arm returns cosine similarity and the keyword arm returns normalised
`ts_rank_cd`, which are not on a common scale and are not even comparable
across queries. RRF uses only the ordering each retriever produced, so it
cannot be skewed by one arm's scores happening to be larger.

Pure functions over ranked lists, deliberately free of any database or
provider dependency, so fusion behaviour can be tested exhaustively.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from core.types import RetrievedChunk, RetrieverContribution, RetrieverKind

# The constant from the original RRF paper (Cormack et al., 2009). It damps the
# influence of the top ranks: with k=60 the gap between rank 1 and rank 2 is
# small, so one retriever being confident does not let it dictate the result.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one ranking, preserving each arm's contribution.

    Score for a chunk is the sum over retrievers of `weight / (k + rank)`,
    with rank 1-based. A chunk found by both arms therefore outranks one found
    by either alone, which is the behaviour hybrid retrieval is there for.

    `weights` defaults to equal weighting; Phase 11 varies it per query.
    Ordering is deterministic: ties break on chunk ID, never on input order.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[UUID, float] = defaultdict(float)
    contributions: dict[UUID, list[RetrieverContribution]] = defaultdict(list)
    exemplar: dict[UUID, RetrievedChunk] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for position, hit in enumerate(ranking):
            chunk_id = hit.provenance.chunk_id
            # Trust the list's order rather than the hit's own `rank` field, so
            # a retriever that mislabels its ranks cannot corrupt the fusion.
            scores[chunk_id] += weight / (k + position + 1)
            contributions[chunk_id].append(
                RetrieverContribution(retriever=hit.retriever, rank=position, score=hit.score)
            )
            # Keep the first sighting: every arm returns the same chunk text
            # and provenance, so this only decides which copy is reused.
            exemplar.setdefault(chunk_id, hit)

    ordered = sorted(
        exemplar,
        key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)),
    )

    fused: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered):
        hit = exemplar[chunk_id]
        found_by = contributions[chunk_id]
        fused.append(
            RetrievedChunk(
                text=hit.text,
                # The fusion score is the ranking signal; the per-retriever
                # scores live on the contributions, where they stay meaningful.
                score=scores[chunk_id],
                retriever=RetrieverKind.HYBRID,
                provenance=hit.provenance,
                rank=rank,
                # Matched terms only ever come from the lexical arm.
                matched_terms=_first_matched_terms(rankings, chunk_id),
                fusion_score=scores[chunk_id],
                contributions=tuple(found_by),
                metadata=dict(hit.metadata),
            )
        )
    return fused


def _first_matched_terms(rankings: list[list[RetrievedChunk]], chunk_id: UUID) -> tuple[str, ...]:
    """Matched terms for a chunk, from whichever arm reported any."""
    for ranking in rankings:
        for hit in ranking:
            if hit.provenance.chunk_id == chunk_id and hit.matched_terms:
                return hit.matched_terms
    return ()


def weighted_score_fusion(
    rankings: list[list[RetrievedChunk]], weights: list[float]
) -> list[RetrievedChunk]:
    """Alpha-weighted fusion over normalised scores.

    Not implemented: it requires calibrating cosine similarity against
    `ts_rank_cd`, which are on different scales and, for the keyword arm, not
    comparable across queries. Doing that honestly needs the score
    distributions the Phase 9 evaluation harness produces, so this arrives with
    adaptive retrieval in Phase 11 rather than being guessed at now.
    """
    raise NotImplementedError
