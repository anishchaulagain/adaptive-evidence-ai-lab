"""Ambient request/execution context.

Carries the identifiers the spec requires on every log line and trace
(section 48) without threading them through every function signature.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from uuid import UUID

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
user_id_var: ContextVar[UUID | None] = ContextVar("user_id", default=None)
organization_id_var: ContextVar[UUID | None] = ContextVar("organization_id", default=None)
project_id_var: ContextVar[UUID | None] = ContextVar("project_id", default=None)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Snapshot of the ambient context, for logging and trace enrichment."""

    request_id: str | None = None
    trace_id: str | None = None
    user_id: UUID | None = None
    organization_id: UUID | None = None
    project_id: UUID | None = None


def current_context() -> RequestContext:
    return RequestContext(
        request_id=request_id_var.get(),
        trace_id=trace_id_var.get(),
        user_id=user_id_var.get(),
        organization_id=organization_id_var.get(),
        project_id=project_id_var.get(),
    )


@contextmanager
def bind_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    user_id: UUID | None = None,
    organization_id: UUID | None = None,
    project_id: UUID | None = None,
) -> Iterator[None]:
    """Bind identifiers for the duration of a block, restoring them afterwards.

    Used by HTTP middleware and by background workers, which have no request.
    """
    tokens: list[tuple[ContextVar[object], Token[object]]] = []
    pairs: list[tuple[ContextVar[object], object]] = [
        (request_id_var, request_id),  # type: ignore[list-item]
        (trace_id_var, trace_id),  # type: ignore[list-item]
        (user_id_var, user_id),  # type: ignore[list-item]
        (organization_id_var, organization_id),  # type: ignore[list-item]
        (project_id_var, project_id),  # type: ignore[list-item]
    ]
    for var, value in pairs:
        if value is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
