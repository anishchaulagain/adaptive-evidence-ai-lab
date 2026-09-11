"""Version 1 router aggregation.

Endpoint surface follows spec section 44. Mounted by `create_app()` under
`settings.API_V1_PREFIX`.
"""

from fastapi import APIRouter

from app.api.v1.routes import (
    benchmarks,
    documents,
    evaluations,
    experiments,
    health,
    models,
    projects,
    query,
    sources,
    traces,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(sources.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(traces.router)
api_router.include_router(evaluations.router)
api_router.include_router(experiments.router)
api_router.include_router(benchmarks.router)
api_router.include_router(models.router)
