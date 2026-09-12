"""Worker process lifecycle.

The worker runs outside the API's lifespan, so it must establish its own
resources. Tests that invoke a job through the API fixtures cannot catch a
gap here — they inherit the API's initialisation.
"""

from __future__ import annotations

import pytest

from app.db.session import get_sessionmaker
from app.workers.queue import get_queue
from app.workers.settings import WorkerSettings, shutdown, startup

pytestmark = pytest.mark.integration


async def test_worker_startup_establishes_database_and_queue() -> None:
    """Ingestion chains an embedding job, so a worker without a queue pool
    fails *after* doing all the parsing work."""
    ctx: dict[str, object] = {}
    await startup(ctx)
    try:
        assert ctx["settings"] is not None
        assert get_sessionmaker() is not None
        assert get_queue() is not None
    finally:
        await shutdown(ctx)


async def test_every_task_is_registered() -> None:
    """A task missing from the registry is silently unrunnable."""
    names = {function.__name__ for function in WorkerSettings.functions}

    assert "ingest_document" in names
    assert "embed_document_chunks" in names
