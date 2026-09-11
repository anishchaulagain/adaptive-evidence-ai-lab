"""Liveness and readiness probes.

`/health` stays dependency-free so it still answers while Postgres is down —
that distinction is what makes it usable as a container liveness probe.
`/ready` is the one that reports on dependencies.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import SettingsDep
from app.db.session import check_database
from app.schemas.common import HealthResponse, ReadinessResponse
from app.workers.queue import check_queue

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness: the process is up and serving."""
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        environment=str(settings.ENVIRONMENT),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def ready(response: Response) -> ReadinessResponse:
    """Readiness: every dependency is reachable.

    Returns 503 when degraded so orchestrators stop routing traffic here.
    """
    checks = {
        "database": await check_database(),
        "redis": await check_queue(),
    }
    healthy = all(checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if healthy else "degraded", checks=checks)
