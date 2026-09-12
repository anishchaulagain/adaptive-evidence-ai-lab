"""Retrieval strategy comparison payload (spec section 37.2).

Answers the questions the spec poses directly: did semantic retrieval miss
something, did keyword recover it, and did fusion change the ranking? Those are
questions about *disagreement*, so the payload is organised around each chunk's
rank under every strategy rather than around three separate result lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """One strategy's ranked output."""

    strategy: str
    latency_ms: float
    # (chunk_id, text, score) in rank order.
    hits: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One chunk, and where each strategy placed it."""

    chunk_id: str
    text: str
    # Rank per strategy; None means that strategy did not return it at all.
    ranks: dict[str, int | None] = field(default_factory=dict)
    scores: dict[str, float | None] = field(default_factory=dict)
    found_by: list[str] = field(default_factory=list)


def build_comparison(results: list[StrategyResult]) -> dict[str, Any]:
    """Compare ranked lists chunk by chunk.

    Rows are ordered by the best rank any strategy gave a chunk, so the results
    that matter to at least one strategy appear first — sorting by a single
    strategy's ranking would bury exactly the disagreements being looked for.
    """
    strategies = [result.strategy for result in results]
    texts: dict[str, str] = {}
    ranks: dict[str, dict[str, int | None]] = {}
    scores: dict[str, dict[str, float | None]] = {}

    for result in results:
        for rank, (chunk_id, text, score) in enumerate(result.hits):
            texts.setdefault(chunk_id, text)
            ranks.setdefault(chunk_id, dict.fromkeys(strategies))
            scores.setdefault(chunk_id, dict.fromkeys(strategies))
            ranks[chunk_id][result.strategy] = rank
            scores[chunk_id][result.strategy] = score

    rows = [
        ComparisonRow(
            chunk_id=chunk_id,
            text=texts[chunk_id],
            ranks=ranks[chunk_id],
            scores=scores[chunk_id],
            found_by=[s for s in strategies if ranks[chunk_id][s] is not None],
        )
        for chunk_id in texts
    ]
    rows.sort(
        key=lambda row: (
            min((r for r in row.ranks.values() if r is not None), default=10**6),
            row.chunk_id,
        )
    )

    return {
        "strategies": [
            {
                "strategy": result.strategy,
                "latency_ms": result.latency_ms,
                "result_count": len(result.hits),
            }
            for result in results
        ],
        "rows": [
            {
                "chunk_id": row.chunk_id,
                "text": row.text,
                "ranks": row.ranks,
                "scores": row.scores,
                "found_by": row.found_by,
            }
            for row in rows
        ],
        "overlap": _overlap(rows, strategies),
    }


def _overlap(rows: list[ComparisonRow], strategies: list[str]) -> dict[str, Any]:
    """How much the strategies agreed.

    `unique_to` is the interesting number: a strategy contributing nothing of
    its own is one that could be removed without changing the result.
    """
    unique_to = dict.fromkeys(strategies, 0)
    for row in rows:
        if len(row.found_by) == 1:
            unique_to[row.found_by[0]] += 1

    all_strategies = [row for row in rows if len(row.found_by) == len(strategies)]
    return {
        "total_chunks": len(rows),
        "found_by_all": len(all_strategies),
        "unique_to": unique_to,
    }
