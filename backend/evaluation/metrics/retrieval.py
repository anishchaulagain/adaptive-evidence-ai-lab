"""Retrieval metrics (spec sections 26, 74).

Pure functions over ranked chunk IDs and a gold set. No database, no provider,
no framework — so they can be tested exhaustively and reused from benchmark
and experiment runners.

Every metric returns `None` rather than `0.0` when it is undefined for an
item — an item with no gold chunks has no recall, and averaging a fabricated
zero into a benchmark would understate the system under test. Aggregation
skips `None` and reports how many items actually contributed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID


def _top(retrieved: Sequence[UUID], k: int) -> list[UUID]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(retrieved[:k])


def recall_at_k(retrieved: Sequence[UUID], gold: set[UUID], k: int) -> float | None:
    """Fraction of gold chunks that appear in the top k.

    Undefined when nothing is gold: there is nothing to recall.
    """
    if not gold:
        return None
    found = len(set(_top(retrieved, k)) & gold)
    return found / len(gold)


def precision_at_k(retrieved: Sequence[UUID], gold: set[UUID], k: int) -> float | None:
    """Fraction of the top k that is gold.

    Divided by how many results there actually are, not by k. Dividing by k
    would penalise a system for a corpus that holds fewer than k chunks, which
    measures the corpus rather than the retriever.
    """
    if not gold:
        return None
    top = _top(retrieved, k)
    if not top:
        return 0.0
    return len(set(top) & gold) / len(top)


def hit_rate_at_k(retrieved: Sequence[UUID], gold: set[UUID], k: int) -> float | None:
    """1.0 if any gold chunk is in the top k, else 0.0."""
    if not gold:
        return None
    return 1.0 if set(_top(retrieved, k)) & gold else 0.0


def reciprocal_rank(retrieved: Sequence[UUID], gold: set[UUID]) -> float | None:
    """1 / (rank of the first gold chunk), rank being 1-based.

    0.0 when no gold chunk was retrieved at all — that is a real miss, not an
    undefined measurement.
    """
    if not gold:
        return None
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / position
    return 0.0


def dcg_at_k(retrieved: Sequence[UUID], gold: set[UUID], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    return sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(_top(retrieved, k), start=1)
        if chunk_id in gold
    )


def ndcg_at_k(retrieved: Sequence[UUID], gold: set[UUID], k: int) -> float | None:
    """Normalised DCG: how close the ranking is to the best possible one.

    The ideal ranking puts every gold chunk first, but no more of them than k
    or than the gold set holds — so a system is never penalised for gold it
    could not have fitted into k slots.
    """
    if not gold:
        return None
    ideal_hits = min(len(gold), k)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    if ideal == 0.0:  # pragma: no cover - k >= 1 and gold non-empty
        return None
    return dcg_at_k(retrieved, gold, k) / ideal


def mean(values: Sequence[float | None]) -> float | None:
    """Average, ignoring undefined entries.

    Returns None when nothing was defined, so an empty measurement is reported
    as absent rather than as zero.
    """
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def defined_count(values: Sequence[float | None]) -> int:
    """How many items actually contributed to an aggregate."""
    return sum(1 for value in values if value is not None)
