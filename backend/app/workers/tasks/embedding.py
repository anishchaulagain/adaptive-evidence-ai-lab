"""Embedding generation jobs (spec section 68).

Separate from ingestion so a provider outage retries only the embedding stage
— parsing and chunking results are already durable and need not be redone.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.session import get_sessionmaker
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


async def embed_document_chunks(ctx: dict[str, Any], document_id: UUID) -> int:
    """Embed a document's outstanding chunks in batches.

    Returns the number written, which arq records as the job result.
    """
    settings: Settings = ctx.get("settings") or get_settings()
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        service = EmbeddingService(session, settings)
        try:
            return await service.embed_document(document_id)
        except NotFoundError:
            # Deleted while queued; retrying cannot succeed.
            logger.info("embedding.document_missing", document_id=str(document_id))
            return 0
        finally:
            await service.aclose()
