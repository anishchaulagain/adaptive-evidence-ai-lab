"""Embedding orchestration (spec section 68, Phase 3).

Embedding is the slow, billable stage, so it runs in a worker and is resumable:
only chunks that are missing a vector, or were embedded by a different model,
are sent to the provider.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.providers import get_embedding_pipeline
from app.models.chunk import Chunk
from app.models.document import Document, IngestionStatus
from app.services.base import Service
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from core.errors import DomainError, ErrorCode

logger = get_logger(__name__)


class EmbeddingService(Service):
    """Embeds a document's chunks and records which model produced them."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        pipeline: BatchedEmbeddingPipeline | None = None,
    ) -> None:
        super().__init__(session)
        self.settings = settings
        self._pipeline = pipeline
        self._owns_pipeline = pipeline is None

    async def _get_pipeline(self) -> BatchedEmbeddingPipeline:
        if self._pipeline is None:
            self._pipeline = get_embedding_pipeline(self.settings)
        return self._pipeline

    async def aclose(self) -> None:
        if self._owns_pipeline and self._pipeline is not None:
            await self._pipeline.aclose()
            self._pipeline = None

    async def embed_document(self, document_id: UUID) -> int:
        """Embed a document's outstanding chunks; returns how many were written.

        A failure is recorded on the document with its error code, as ingestion
        does, so the state is inspectable through the API rather than only in
        worker logs.
        """
        document = await self.session.get(Document, document_id)
        if document is None:
            raise NotFoundError("Document not found.", details={"document_id": str(document_id)})

        try:
            pipeline = await self._get_pipeline()
            pending = await self._pending_chunks(document_id, pipeline.model_id)
            if not pending:
                logger.info("embedding.skipped", document_id=str(document_id))
                return 0

            vectors = await pipeline.embed_texts([chunk.text for chunk in pending])
            if len(vectors) != len(pending):  # pragma: no cover - provider contract
                raise DomainError(
                    "The provider returned a mismatched number of embeddings.",
                    code=ErrorCode.EMBEDDING_FAILED,
                )

            for chunk, vector in zip(pending, vectors, strict=True):
                chunk.embedding = vector
                chunk.embedding_model = pipeline.model_id

            document.error_code = None
            document.error_message = None
            await self.session.commit()

            logger.info(
                "embedding.completed",
                document_id=str(document_id),
                embedded=len(pending),
                model=pipeline.model_id,
            )
            return len(pending)
        except DomainError as exc:
            await self.session.rollback()
            await self._record_failure(document_id, exc)
            logger.warning(
                "embedding.failed",
                document_id=str(document_id),
                code=str(exc.code),
            )
            raise

    async def _pending_chunks(self, document_id: UUID, model_id: str) -> list[Chunk]:
        """Chunks with no vector, or one from a different model.

        Re-embedding after a model change is therefore just a re-run, and a
        partially embedded document resumes instead of starting over.
        """
        result = await self.session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                (Chunk.embedding.is_(None)) | (Chunk.embedding_model != model_id),
            )
            .order_by(Chunk.ordinal)
        )
        return list(result)

    async def _record_failure(self, document_id: UUID, error: DomainError) -> None:
        document = await self.session.get(Document, document_id)
        if document is None:  # pragma: no cover - defensive
            return
        document.ingestion_status = IngestionStatus.FAILED
        document.error_code = str(error.code)
        document.error_message = error.message
        await self.session.commit()
