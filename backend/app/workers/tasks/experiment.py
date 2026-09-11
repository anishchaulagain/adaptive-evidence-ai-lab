"""Experiment jobs (spec sections 31-34)."""

from __future__ import annotations

from typing import Any
from uuid import UUID


async def run_experiment(ctx: dict[str, Any], experiment_id: UUID) -> None:
    """Execute each experimental condition under a fixed seed."""
    raise NotImplementedError
