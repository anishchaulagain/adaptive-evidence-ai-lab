"""Document ingestion jobs (spec section 10).

The job is a thin adapter: it opens a session and delegates to
`IngestionService`, so the pipeline is testable without running a worker.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.providers import embeddings_enabled
from app.db.session import get_sessionmaker
from app.models.document import IngestionStatus
from app.services.ingestion_service import IngestionService
from app.workers.queue import enqueue

logger = get_logger(__name__)


async def ingest_document(ctx: dict[str, Any], document_id: UUID) -> None:
    """Parse and chunk a single document, then queue it for embedding.

    Failures are recorded against the document with an explicit error code
    (`PARSING_FAILED`, `INGESTION_FAILED`, ...) and surfaced through the API.
    """
    settings: Settings = ctx.get("settings") or get_settings()
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            document = await IngestionService(session, settings).run_ingestion(document_id)
        except NotFoundError:
            # The document, or its project, was deleted while the job sat in
            # the queue. Retrying can never succeed, so this is terminal rather
            # than a failure worth five more attempts.
            logger.info("ingestion.document_missing", document_id=str(document_id))
            return

    if document.ingestion_status != IngestionStatus.COMPLETED:
        return

    # Embedding is a separate job so a provider outage retries only that stage.
    if embeddings_enabled(settings):
        await enqueue("embed_document_chunks", document_id)
    else:
        # Not an error: without a key the platform still ingests and chunks,
        # and the document becomes dense-retrievable once a key is configured
        # and the job is re-run.
        logger.info("embedding.not_configured", document_id=str(document_id))


async def sync_source(ctx: dict[str, Any], source_id: UUID) -> None:
    """Pull new or changed documents from a connected source.

    Unimplemented: external source connectors are not part of this phase, so
    documents arrive only through direct upload.
    """
    raise NotImplementedError
