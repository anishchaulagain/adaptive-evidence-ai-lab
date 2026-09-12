"""arq worker configuration.

Run with:  arq app.workers.settings.WorkerSettings

Long-running operations (ingestion, OCR, embedding, indexing, evaluations,
experiments, benchmarks) run here so HTTP requests never block
(spec section 46).
"""

from __future__ import annotations

from typing import Any, ClassVar

from arq.connections import RedisSettings

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import dispose_engine, init_engine
from app.workers.queue import close_queue, init_queue
from app.workers.tasks import TASK_FUNCTIONS


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings)
    init_engine(settings)
    # The worker enqueues follow-up jobs of its own (ingestion chains
    # embedding), so it needs its own pool. The API lifespan does not run here.
    await init_queue(settings)
    ctx["settings"] = settings


async def shutdown(ctx: dict[str, Any]) -> None:
    await close_queue()
    await dispose_engine()


def _redis_settings(settings: Settings | None = None) -> RedisSettings:
    settings = settings or get_settings()
    return RedisSettings.from_dsn(str(settings.REDIS_URL))


class WorkerSettings:
    """Entry point discovered by the `arq` CLI."""

    functions: ClassVar[list[Any]] = TASK_FUNCTIONS
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    max_jobs = 10
    job_timeout = 3600  # benchmark and evaluation runs are legitimately long
    keep_result = 86400
