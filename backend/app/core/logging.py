"""Structured logging setup.

Logs are structured (spec rule 14) and must never contain secrets
(spec section 48). `console` rendering is for local development; production
emits JSON for log aggregation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import LogFormat, Settings
from app.core.context import current_context

# Keys that are redacted if any processor or call site tries to log them.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "secret_key",
        "token",
        "access_key",
        "database_url",
        "redis_url",
    }
)
_REDACTED = "[redacted]"


def _redact_secrets(
    _logger: object, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Defence in depth: never let a credential reach the log sink."""
    for key in list(event_dict):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = _REDACTED
    return event_dict


def _bind_request_context(
    _logger: object, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Attach request/trace/user/project identifiers to every log line."""
    ctx = current_context()
    for key, value in (
        ("request_id", ctx.request_id),
        ("trace_id", ctx.trace_id),
        ("user_id", ctx.user_id),
        ("organization_id", ctx.organization_id),
        ("project_id", ctx.project_id),
    ):
        if value is not None and key not in event_dict:
            event_dict[key] = str(value)
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Configure structlog and route stdlib logging through it. Idempotent."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.LOG_FORMAT is LogFormat.JSON
        else structlog.dev.ConsoleRenderer()
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _bind_request_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
