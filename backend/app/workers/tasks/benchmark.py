"""AE-Bench jobs (spec section 29)."""

from __future__ import annotations

from typing import Any
from uuid import UUID


async def run_benchmark(ctx: dict[str, Any], run_id: UUID) -> None:
    """Execute a benchmark suite and record reproducibility metadata."""
    raise NotImplementedError
