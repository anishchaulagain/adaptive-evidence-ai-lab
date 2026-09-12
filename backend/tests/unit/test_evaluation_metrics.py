"""Evaluation metric correctness (spec sections 26, 74).

These numbers are the platform's output — every later conclusion about whether
hybrid beats dense rests on them being right. They are checked against
hand-computed values rather than against the implementation's own behaviour.
"""

from __future__ import annotations

import math
from uuid import UUID

import pytest

from evaluation.metrics.citation import (
    ClaimCitations,
    citation_precision,
    citation_recall,
    citation_validity,
    evidence_coverage,
    unsupported_claim_rate,
)
from evaluation.metrics.retrieval import (
    defined_count,
    hit_rate_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit


def _id(n: int) -> UUID:
    return UUID(int=n)


A, B, C, D, E = (_id(n) for n in range(1, 6))


# --- recall ---------------------------------------------------------------


def test_recall_counts_gold_found_in_the_top_k() -> None:
    assert recall_at_k([A, B, C], {A, D}, k=3) == pytest.approx(0.5)


def test_recall_ignores_results_below_k() -> None:
    assert recall_at_k([A, B, C], {C}, k=2) == 0.0
    assert recall_at_k([A, B, C], {C}, k=3) == 1.0


def test_recall_is_undefined_without_gold() -> None:
    """An item with no gold chunks has no recall. Averaging a fabricated zero
    into a benchmark would understate the system under test."""
    assert recall_at_k([A, B], set(), k=2) is None


def test_recall_handles_fewer_results_than_k() -> None:
    assert recall_at_k([A], {A}, k=10) == 1.0


# --- precision ------------------------------------------------------------


def test_precision_divides_by_results_not_by_k() -> None:
    """Dividing by k would penalise a retriever for a corpus holding fewer
    than k chunks, which measures the corpus rather than the retriever."""
    assert precision_at_k([A, B], {A}, k=10) == pytest.approx(0.5)


def test_precision_of_a_perfect_ranking_is_one() -> None:
    assert precision_at_k([A, B], {A, B}, k=2) == 1.0


def test_precision_with_no_results_is_zero() -> None:
    assert precision_at_k([], {A}, k=5) == 0.0


# --- hit rate and MRR -----------------------------------------------------


@pytest.mark.parametrize(
    ("retrieved", "k", "expected"),
    [([A, B, C], 3, 1.0), ([A, B, C], 1, 0.0), ([], 3, 0.0)],
)
def test_hit_rate(retrieved: list[UUID], k: int, expected: float) -> None:
    assert hit_rate_at_k(retrieved, {C}, k=k) == expected


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [([A, B, C], 1.0), ([B, A, C], 0.5), ([B, C, A], 1 / 3), ([B, C], 0.0)],
)
def test_reciprocal_rank_uses_the_first_gold_hit(retrieved: list[UUID], expected: float) -> None:
    assert reciprocal_rank(retrieved, {A}) == pytest.approx(expected)


def test_reciprocal_rank_of_a_total_miss_is_zero_not_undefined() -> None:
    """Retrieving no gold is a real miss, not an unmeasurable item."""
    assert reciprocal_rank([B, C], {A}) == 0.0


def test_reciprocal_rank_is_undefined_without_gold() -> None:
    assert reciprocal_rank([A], set()) is None


# --- nDCG -----------------------------------------------------------------


def test_ndcg_of_a_perfect_ranking_is_one() -> None:
    assert ndcg_at_k([A, B, C], {A, B}, k=3) == pytest.approx(1.0)


def test_ndcg_penalises_a_worse_ordering() -> None:
    perfect = ndcg_at_k([A, B, C], {A, B}, k=3)
    worse = ndcg_at_k([C, A, B], {A, B}, k=3)
    assert perfect is not None and worse is not None
    assert worse < perfect


def test_ndcg_matches_the_hand_computed_value() -> None:
    """Gold at ranks 2 and 3; ideal has them at 1 and 2."""
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)

    assert ndcg_at_k([C, A, B], {A, B}, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_does_not_penalise_gold_that_cannot_fit_in_k() -> None:
    """With 5 gold chunks and k=1, retrieving one of them at rank 1 is the best
    achievable result and must score 1.0."""
    assert ndcg_at_k([A], {A, B, C, D, E}, k=1) == pytest.approx(1.0)


def test_ndcg_of_a_total_miss_is_zero() -> None:
    assert ndcg_at_k([D, E], {A}, k=2) == 0.0


# --- aggregation ----------------------------------------------------------


def test_mean_skips_undefined_items() -> None:
    assert mean([1.0, None, 0.0]) == pytest.approx(0.5)


def test_mean_of_nothing_defined_is_none() -> None:
    """Reported as absent rather than as zero."""
    assert mean([None, None]) is None
    assert mean([]) is None


def test_defined_count_reports_how_many_items_contributed() -> None:
    assert defined_count([1.0, None, 0.0, None]) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k([A], {A}, k=k)


# --- citation metrics -----------------------------------------------------


def test_citation_precision_measures_padding() -> None:
    """Citing extra passages reads as well-sourced while not being."""
    assert citation_precision({A, B}, {A}) == pytest.approx(0.5)


def test_citation_recall_measures_gold_actually_cited() -> None:
    assert citation_recall({A}, {A, B}) == pytest.approx(0.5)


def test_citation_metrics_are_undefined_without_gold() -> None:
    assert citation_precision({A}, set()) is None
    assert citation_recall({A}, set()) is None


def test_citation_precision_is_undefined_when_nothing_was_cited() -> None:
    """No citations means no precision to measure; the gap is captured by
    unsupported claim rate instead."""
    assert citation_precision(set(), {A}) is None


def test_evidence_coverage_reports_how_much_retrieval_was_used() -> None:
    assert evidence_coverage({A}, [A, B, C, D]) == pytest.approx(0.25)


def test_evidence_coverage_deduplicates_retrieved_chunks() -> None:
    assert evidence_coverage({A}, [A, A, B]) == pytest.approx(0.5)


def test_unsupported_claim_rate() -> None:
    claims = [
        ClaimCitations("cited", (A,)),
        ClaimCitations("uncited", ()),
        ClaimCitations("also uncited", ()),
    ]

    assert unsupported_claim_rate(claims) == pytest.approx(2 / 3)


def test_unsupported_claim_rate_of_no_claims_is_undefined() -> None:
    assert unsupported_claim_rate([]) is None


def test_citation_validity_detects_invented_citations() -> None:
    """Below 1.0 means the generator cited a passage that was never retrieved."""
    claims = [ClaimCitations("one", (A, E))]

    assert citation_validity(claims, [A, B, C]) == pytest.approx(0.5)


def test_citation_validity_is_one_when_every_citation_is_real() -> None:
    claims = [ClaimCitations("one", (A,)), ClaimCitations("two", (B,))]

    assert citation_validity(claims, [A, B, C]) == 1.0


def test_citation_validity_is_undefined_without_citations() -> None:
    assert citation_validity([ClaimCitations("x", ())], [A]) is None
