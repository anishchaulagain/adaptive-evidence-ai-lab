"""Index maintenance jobs (vector index and full-text index)."""

from __future__ import annotations

from typing import Any
from uuid import UUID


async def reindex_project(ctx: dict[str, Any], project_id: UUID) -> None:
    """Rebuild vector and keyword indexes for a project."""
    raise NotImplementedError
