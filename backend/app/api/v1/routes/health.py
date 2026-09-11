"""Liveness and readiness probes.

`/health` must stay dependency-free so it can answer while the database is
down; `/ready` is the one that checks dependencies.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import SettingsDep
from app.schemas.common import HealthResponse, ReadinessResponse

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
async def ready() -> ReadinessResponse:
    """Readiness: database, cache and queue are reachable."""
    raise NotImplementedError
