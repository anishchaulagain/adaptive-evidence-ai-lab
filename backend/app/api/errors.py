"""Translate exceptions into a single, stable error envelope.

Clients (and the trace viewer) always receive the same shape:

    {"error": {"code": "...", "message": "...", "details": {...},
               "request_id": "...", "trace_id": "..."}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import Settings
from app.core.context import current_context
from app.core.errors import DomainError, ErrorCode, status_code_for
from app.core.logging import get_logger

logger = get_logger(__name__)


def _envelope(
    code: ErrorCode | str,
    message: str,
    *,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    ctx = current_context()
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": str(code),
                "message": message,
                "details": details or {},
                "request_id": ctx.request_id,
                "trace_id": ctx.trace_id,
            }
        },
    )


def register_exception_handlers(app: FastAPI, settings: Settings) -> None:
    """Install handlers on the application. Called from `create_app()`."""

    @app.exception_handler(DomainError)
    async def _domain_error(_request: Request, exc: DomainError) -> JSONResponse:
        """Handles `AppError` and anything raised from `core` alike, so an
        error escaping a pipeline stage still gets a code, not a bare 500."""
        status_code = status_code_for(exc)
        log = logger.warning if status_code < 500 else logger.error
        log(
            "app.error",
            code=str(exc.code),
            message=exc.message,
            status_code=status_code,
        )
        return _envelope(exc.code, exc.message, status_code=status_code, details=exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(
            ErrorCode.VALIDATION_ERROR,
            "Request validation failed.",
            status_code=422,
            details={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INTERNAL_ERROR
        return _envelope(code, str(exc.detail), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        # Log the traceback; never leak internals to the client in production.
        logger.exception("unhandled.exception")
        message = str(exc) if settings.DEBUG else "An internal error occurred."
        return _envelope(ErrorCode.INTERNAL_ERROR, message, status_code=500)
