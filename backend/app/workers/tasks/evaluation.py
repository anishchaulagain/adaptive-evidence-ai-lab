"""Evaluation run jobs (spec section 26)."""

from __future__ import annotations

from typing import Any
from uuid import UUID


async def run_evaluation(ctx: dict[str, Any], run_id: UUID) -> None:
    """Execute an evaluation run and persist real, computed metrics."""
    raise NotImplementedError
