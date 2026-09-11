"""Document ingestion jobs (spec section 10).

The job is a thin adapter: it opens a session and delegates to
`IngestionService`, so the pipeline is testable without running a worker.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core.config import Settings, get_settings
from app.db.session import get_sessionmaker
from app.services.ingestion_service import IngestionService


async def ingest_document(ctx: dict[str, Any], document_id: UUID) -> None:
    """Parse and chunk a single document.

    Failures are recorded against the document with an explicit error code
    (`PARSING_FAILED`, `INGESTION_FAILED`, ...) and surfaced through the API.
    """
    settings: Settings = ctx.get("settings") or get_settings()
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await IngestionService(session, settings).run_ingestion(document_id)


async def sync_source(ctx: dict[str, Any], source_id: UUID) -> None:
    """Pull new or changed documents from a connected source.

    Unimplemented: external source connectors are not part of this phase, so
    documents arrive only through direct upload.
    """
    raise NotImplementedError
