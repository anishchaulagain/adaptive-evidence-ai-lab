"""Assign a request ID and bind it to the logging/trace context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

Handler = Callable[[Request], Awaitable[Response]]


async def request_context_middleware(request: Request, call_next: Handler) -> Response:
    """Accept an inbound `X-Request-ID` or generate one, and echo it back.

    Bound for the lifetime of the request so every log line and trace span
    carries it (spec section 48).
    """
    # Imported lazily so middleware import order never pulls in settings.
    from app.core.context import bind_context

    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
    with bind_context(request_id=request_id):
        response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response
