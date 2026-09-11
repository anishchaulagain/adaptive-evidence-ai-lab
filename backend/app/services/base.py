"""Service layer conventions.

Services orchestrate: they own transactions and call into `core.*` pipelines
and `app.models` persistence. They never import FastAPI — that keeps them
usable from background workers and from experiment scripts.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


class Service:
    """Base class carrying the unit of work."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
