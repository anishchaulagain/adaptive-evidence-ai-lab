"""FastAPI application factory.

Kept free of business logic: it wires configuration, logging, middleware,
routers and error handlers, and owns startup/shutdown of shared resources.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.errors import register_exception_handlers
from app.api.v1.router import api_router
from app.core.config import AuthMode, Settings, get_settings, validate_runtime_settings
from app.core.logging import configure_logging, get_logger
from app.db.bootstrap import ensure_dev_principal
from app.db.session import dispose_engine, get_sessionmaker, init_engine
from app.middleware.access_log import access_log_middleware
from app.middleware.request_context import request_context_middleware
from app.workers.queue import close_queue, init_queue

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the lifecycle of shared resources (database, cache, queue)."""
    settings: Settings = app.state.settings
    init_engine(settings)
    await init_queue(settings)

    if settings.AUTH_MODE is AuthMode.DISABLED:
        await ensure_dev_principal(get_sessionmaker())

    logger.info(
        "application.startup",
        environment=str(settings.ENVIRONMENT),
        version=settings.VERSION,
        auth_mode=str(settings.AUTH_MODE),
    )
    try:
        yield
    finally:
        await close_queue()
        await dispose_engine()
        logger.info("application.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Tests call this with overridden settings."""
    settings = settings or get_settings()
    validate_runtime_settings(settings)
    configure_logging(settings)

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan,
        # OpenAPI is served only outside production (spec section 42).
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings

    # Middleware runs bottom-up: request context is outermost.
    app.add_middleware(BaseHTTPMiddleware, dispatch=access_log_middleware)
    app.add_middleware(BaseHTTPMiddleware, dispatch=request_context_middleware)
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    register_exception_handlers(app, settings)
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()
