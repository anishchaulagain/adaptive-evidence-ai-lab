"""Reciprocal Rank Fusion invariants.

Fusion is a pure function over ranked lists, so its behaviour is pinned down
here rather than inferred from end-to-end retrieval results.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from core.retrieval.fusion import RRF_K, reciprocal_rank_fusion, weighted_score_fusion
from core.types import Provenance, RetrievedChunk, RetrieverKind

pytestmark = pytest.mark.unit

DOCUMENT_ID = UUID("00000000-0000-0000-0000-0000000000aa")


def _hit(
    chunk_id: UUID,
    rank: int,
    *,
    retriever: RetrieverKind,
    score: float = 0.5,
    matched_terms: tuple[str, ...] = (),
) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"chunk {chunk_id}",
        score=score,
        retriever=retriever,
        provenance=Provenance(document_id=DOCUMENT_ID, chunk_id=chunk_id),
        rank=rank,
        matched_terms=matched_terms,
    )


def _ranking(ids: list[UUID], retriever: RetrieverKind) -> list[RetrievedChunk]:
    return [_hit(cid, rank, retriever=retriever) for rank, cid in enumerate(ids)]


def _ids(hits: list[RetrievedChunk]) -> list[UUID]:
    return [hit.provenance.chunk_id for hit in hits]


# --- ranking behaviour ----------------------------------------------------


def test_a_chunk_found_by_both_arms_outranks_one_found_by_either() -> None:
    """The core promise of hybrid retrieval: agreement beats confidence."""
    both, dense_only, keyword_only = uuid4(), uuid4(), uuid4()

    fused = reciprocal_rank_fusion(
        [
            _ranking([dense_only, both], RetrieverKind.SEMANTIC),
            _ranking([keyword_only, both], RetrieverKind.KEYWORD),
        ]
    )

    assert _ids(fused)[0] == both


def test_a_single_arm_ranking_is_preserved_exactly() -> None:
    """With one non-empty list, fusion is order-preserving — so degrading to
    one arm cannot reshuffle results."""
    ids = [uuid4() for _ in range(5)]

    fused = reciprocal_rank_fusion([_ranking(ids, RetrieverKind.SEMANTIC), []])

    assert _ids(fused) == ids
    assert [hit.rank for hit in fused] == [0, 1, 2, 3, 4]


def test_an_empty_keyword_arm_is_routine() -> None:
    """The keyword arm is conjunctive and often returns nothing; that must
    reproduce the dense ranking rather than fail."""
    ids = [uuid4() for _ in range(3)]

    fused = reciprocal_rank_fusion([_ranking(ids, RetrieverKind.SEMANTIC), []])

    assert _ids(fused) == ids


def test_fusing_nothing_yields_nothing() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_rank_is_recomputed_from_list_position() -> None:
    """Fusion trusts list order, not each hit's self-reported rank, so a
    retriever that mislabels its ranks cannot corrupt the result."""
    first, second = uuid4(), uuid4()
    mislabelled = [
        _hit(first, rank=99, retriever=RetrieverKind.SEMANTIC),
        _hit(second, rank=99, retriever=RetrieverKind.SEMANTIC),
    ]

    fused = reciprocal_rank_fusion([mislabelled, []])

    assert _ids(fused) == [first, second]
    assert [hit.rank for hit in fused] == [0, 1]


def test_scores_follow_the_rrf_formula() -> None:
    top, second = uuid4(), uuid4()

    fused = reciprocal_rank_fusion([_ranking([top, second], RetrieverKind.SEMANTIC), []])

    assert fused[0].fusion_score == pytest.approx(1 / (RRF_K + 1))
    assert fused[1].fusion_score == pytest.approx(1 / (RRF_K + 2))


def test_agreement_sums_across_arms() -> None:
    shared = uuid4()

    fused = reciprocal_rank_fusion(
        [
            _ranking([shared], RetrieverKind.SEMANTIC),
            _ranking([shared], RetrieverKind.KEYWORD),
        ]
    )

    assert fused[0].fusion_score == pytest.approx(2 / (RRF_K + 1))


def test_ordering_is_deterministic_for_tied_scores() -> None:
    """Ties must not depend on input order, or the same query could return
    different rankings between runs."""
    ids = sorted([uuid4() for _ in range(4)], key=str)

    dense = _ranking(ids, RetrieverKind.SEMANTIC)
    keyword = _ranking(list(reversed(ids)), RetrieverKind.KEYWORD)

    forward = reciprocal_rank_fusion([dense, keyword])
    backward = reciprocal_rank_fusion([keyword, dense])

    assert _ids(forward) == _ids(backward)


# --- provenance (spec section 17) -----------------------------------------


def test_contributions_record_every_arm_that_found_the_chunk() -> None:
    shared = uuid4()

    fused = reciprocal_rank_fusion(
        [
            [_hit(shared, 0, retriever=RetrieverKind.SEMANTIC, score=0.87)],
            [_hit(shared, 0, retriever=RetrieverKind.KEYWORD, score=0.12)],
        ]
    )

    hit = fused[0]
    assert {item.retriever for item in hit.contributions} == {
        RetrieverKind.SEMANTIC,
        RetrieverKind.KEYWORD,
    }
    semantic = hit.contribution(RetrieverKind.SEMANTIC)
    keyword = hit.contribution(RetrieverKind.KEYWORD)
    assert semantic is not None and semantic.score == pytest.approx(0.87)
    assert keyword is not None and keyword.score == pytest.approx(0.12)


def test_pre_fusion_rank_is_preserved_per_arm() -> None:
    """ "Did semantic miss this and keyword recover it?" is answerable only if
    each arm's original rank survives fusion."""
    recovered = uuid4()
    dense = _ranking([uuid4() for _ in range(5)] + [recovered], RetrieverKind.SEMANTIC)
    keyword = _ranking([recovered], RetrieverKind.KEYWORD)

    fused = reciprocal_rank_fusion([dense, keyword])

    hit = next(h for h in fused if h.provenance.chunk_id == recovered)
    semantic = hit.contribution(RetrieverKind.SEMANTIC)
    keyword_contribution = hit.contribution(RetrieverKind.KEYWORD)
    assert semantic is not None and semantic.rank == 5
    assert keyword_contribution is not None and keyword_contribution.rank == 0


def test_a_chunk_only_one_arm_found_records_only_that_arm() -> None:
    dense_only = uuid4()

    fused = reciprocal_rank_fusion(
        [_ranking([dense_only], RetrieverKind.SEMANTIC), _ranking([uuid4()], RetrieverKind.KEYWORD)]
    )

    hit = next(h for h in fused if h.provenance.chunk_id == dense_only)
    assert [item.retriever for item in hit.contributions] == [RetrieverKind.SEMANTIC]
    assert hit.contribution(RetrieverKind.KEYWORD) is None


def test_fused_hits_are_labelled_hybrid() -> None:
    fused = reciprocal_rank_fusion([_ranking([uuid4()], RetrieverKind.SEMANTIC), []])

    assert fused[0].retriever is RetrieverKind.HYBRID


def test_matched_terms_survive_from_the_lexical_arm() -> None:
    """Only the keyword arm produces matched terms; fusion must not drop them."""
    shared = uuid4()

    fused = reciprocal_rank_fusion(
        [
            [_hit(shared, 0, retriever=RetrieverKind.SEMANTIC)],
            [_hit(shared, 0, retriever=RetrieverKind.KEYWORD, matched_terms=("err", "5521"))],
        ]
    )

    assert fused[0].matched_terms == ("err", "5521")


# --- weighting and validation ---------------------------------------------


def test_weights_shift_the_ranking() -> None:
    """Phase 11 varies weights per query, so weighting must actually bite."""
    dense_top, keyword_top = uuid4(), uuid4()
    rankings = [
        _ranking([dense_top], RetrieverKind.SEMANTIC),
        _ranking([keyword_top], RetrieverKind.KEYWORD),
    ]

    dense_favoured = reciprocal_rank_fusion(rankings, weights=[2.0, 1.0])
    keyword_favoured = reciprocal_rank_fusion(rankings, weights=[1.0, 2.0])

    assert _ids(dense_favoured)[0] == dense_top
    assert _ids(keyword_favoured)[0] == keyword_top


def test_mismatched_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="one entry per ranking"):
        reciprocal_rank_fusion([[], []], weights=[1.0])


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([[]], k=k)


def test_weighted_score_fusion_is_not_implemented() -> None:
    """Explicitly unimplemented: calibrating cosine against ts_rank_cd needs
    score distributions the Phase 9 harness has not produced yet."""
    with pytest.raises(NotImplementedError):
        weighted_score_fusion([[]], [1.0])
