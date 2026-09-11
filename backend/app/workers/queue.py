"""Job enqueueing from the API process.

The pool is created once in the application lifespan and reused: arq's
`ArqRedis` is a Redis client, so readiness checks share this connection rather
than opening a second one.
"""

from __future__ import annotations

from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import Settings

_pool: ArqRedis | None = None


async def init_queue(settings: Settings) -> ArqRedis:
    """Create the shared arq pool. Called once from the app lifespan."""
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(str(settings.REDIS_URL)))
    return _pool


def get_queue() -> ArqRedis:
    if _pool is None:
        raise RuntimeError("Queue is not initialised; call init_queue() first.")
    return _pool


async def check_queue() -> bool:
    """Readiness probe for Redis."""
    try:
        await get_queue().ping()
    except Exception:
        return False
    return True


async def enqueue(task_name: str, *args: Any, **kwargs: Any) -> str | None:
    """Enqueue a job and return its ID so callers can report status.

    Returns `None` when arq drops the job as a duplicate.
    """
    job = await get_queue().enqueue_job(task_name, *args, **kwargs)
    return job.job_id if job is not None else None


async def close_queue() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
    _pool = None
