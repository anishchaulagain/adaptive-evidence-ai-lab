"""Document ingestion jobs (spec section 10)."""

from __future__ import annotations

from typing import Any
from uuid import UUID


async def ingest_document(ctx: dict[str, Any], document_id: UUID) -> None:
    """Parse -> chunk -> embed -> index a single document.

    Failures are recorded against the document with an explicit error code
    (`INGESTION_FAILED`, `PARSING_FAILED`, ...) and surfaced in the trace.
    """
    raise NotImplementedError


async def sync_source(ctx: dict[str, Any], source_id: UUID) -> None:
    """Pull new or changed documents from a connected source."""
    raise NotImplementedError
