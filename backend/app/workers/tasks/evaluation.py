"""Evaluation run jobs (spec section 26).

Runs live in a worker because a dataset of any size takes far longer than an
HTTP request should (spec section 46).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.session import get_sessionmaker
from app.services.evaluation_service import EvaluationService

logger = get_logger(__name__)


async def run_evaluation(ctx: dict[str, Any], run_id: UUID) -> int:
    """Execute an evaluation run, returning how many items completed."""
    settings: Settings = ctx.get("settings") or get_settings()
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            run = await EvaluationService(session, settings).execute(run_id)
        except NotFoundError:
            # Deleted while queued; retrying cannot succeed.
            logger.info("evaluation.run_missing", run_id=str(run_id))
            return 0
        return run.completed_count
