"""Project lifecycle and tenant-scoped access checks."""

from __future__ import annotations

from app.services.base import Service


class ProjectService(Service):
    """Create, list and authorize projects (spec sections 42, 43)."""
