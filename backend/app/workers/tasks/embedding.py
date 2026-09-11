"""Embedding generation jobs."""

from __future__ import annotations

from typing import Any
from uuid import UUID


async def embed_document_chunks(ctx: dict[str, Any], document_id: UUID) -> None:
    """Embed every chunk of a document in batches, respecting provider limits."""
    raise NotImplementedError
