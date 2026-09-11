"""Job enqueueing from the API process."""

from __future__ import annotations

from typing import Any

from arq import create_pool
from arq.connections import ArqRedis

_pool: ArqRedis | None = None


async def get_queue() -> ArqRedis:
    """Return the shared arq connection pool, creating it on first use."""
    raise NotImplementedError


async def enqueue(task_name: str, *args: Any, **kwargs: Any) -> str:
    """Enqueue a job and return its ID so callers can report status."""
    raise NotImplementedError


async def close_queue() -> None:
    raise NotImplementedError
