"""Evaluation heatmap payload (spec section 37.6).

A matrix of one metric across retrieval strategies and evidence conditions —
the view that makes "hybrid wins on clean but loses on conflicting" visible at
a glance instead of buried in per-run JSON.

Spec section 37 is explicit that its example numbers are placeholders and that
benchmark results must never be fabricated. Nothing here invents a value: a
combination that was never run is `None`, and a viewer must render it as a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visualization.types import Heatmap, HeatmapCell


@dataclass(frozen=True, slots=True)
class RunInput:
    """One completed evaluation run, flattened."""

    run_id: str
    strategy: str
    # {condition: {metric: {"mean": float | None, "n": int}}}
    by_condition: dict[str, dict[str, Any]]


def build_heatmap(runs: list[RunInput], *, metric: str) -> Heatmap:
    """Build a strategy x condition matrix for one metric.

    When several runs share a strategy and condition, the most recent wins —
    callers pass runs newest-first. Averaging them would silently mix different
    configurations into one number.
    """
    rows: list[str] = []
    columns: list[str] = []
    seen: dict[tuple[str, str], HeatmapCell] = {}

    for run in runs:
        if run.strategy not in rows:
            rows.append(run.strategy)
        for condition, metrics in run.by_condition.items():
            if condition not in columns:
                columns.append(condition)
            key = (run.strategy, condition)
            if key in seen:
                continue
            entry = metrics.get(metric) or {}
            seen[key] = HeatmapCell(
                row=run.strategy,
                column=condition,
                value=entry.get("mean"),
                n=int(entry.get("n", 0) or 0),
                run_id=run.run_id,
            )

    rows.sort()
    columns.sort()
    cells = [
        seen.get(
            (row, column),
            # Never run: a gap, not a zero.
            HeatmapCell(row=row, column=column, value=None, n=0, run_id=None),
        )
        for row in rows
        for column in columns
    ]
    return Heatmap(metric=metric, rows=rows, columns=columns, cells=cells)
