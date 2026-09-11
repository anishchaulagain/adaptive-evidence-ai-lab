"""Structured access logging with request latency (spec section 48)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger(__name__)

Handler = Callable[[Request], Awaitable[Response]]


async def access_log_middleware(request: Request, call_next: Handler) -> Response:
    """Emit one structured line per request. Query strings are not logged —
    they may carry user data."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request.failed",
            method=request.method,
            path=request.url.path,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "request.completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    return response
