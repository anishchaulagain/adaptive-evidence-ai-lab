"""Ingestion orchestration (spec section 10).

Upload validates and stores, then hands off: parsing, chunking and (later)
embedding run in a worker, so an HTTP request never blocks on them
(spec section 46).
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
from typing import BinaryIO
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import (
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.core.logging import get_logger
from app.core.security import Principal
from app.core.storage import get_object_storage
from app.models.chunk import Chunk
from app.models.document import Document, IngestionStatus
from app.services.base import Service
from core.chunking.recursive import RecursiveCharacterChunker
from core.errors import DomainError, ErrorCode
from core.parsing.registry import get_parser, supported_mime_types
from core.storage.base import ObjectStorage

logger = get_logger(__name__)

_READ_BLOCK_BYTES = 1024 * 1024

# Browsers report these inconsistently, so resolve them by extension.
_EXTENSION_MIME_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}


def resolve_mime_type(declared: str | None, filename: str | None) -> str:
    """Determine the media type, preferring a known extension over the client.

    Clients misreport uploads routinely (application/octet-stream for Markdown,
    for instance), so a recognised extension wins over the declared type.
    """
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
        if suffix in _EXTENSION_MIME_TYPES:
            return _EXTENSION_MIME_TYPES[suffix]
        guessed, _ = mimetypes.guess_type(filename)
        if guessed is not None and guessed in supported_mime_types():
            return guessed
    if declared:
        return declared.split(";", 1)[0].strip().lower()
    return "application/octet-stream"


class IngestionService(Service):
    """Accepts uploads and runs the document pipeline."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        storage: ObjectStorage | None = None,
    ) -> None:
        super().__init__(session)
        self.settings = settings
        self.storage = storage or get_object_storage(settings)

    # --- upload ----------------------------------------------------------

    async def create_from_upload(
        self,
        *,
        stream: BinaryIO,
        filename: str | None,
        declared_mime_type: str | None,
        project_id: UUID,
        principal: Principal,
    ) -> Document:
        """Validate, store and register a document, then queue its ingestion."""
        mime_type = resolve_mime_type(declared_mime_type, filename)
        if mime_type not in self.settings.ALLOWED_UPLOAD_MIME_TYPES:
            raise UnsupportedMediaTypeError(
                f"{mime_type} cannot be ingested.",
                details={
                    "mime_type": mime_type,
                    "supported": sorted(self.settings.ALLOWED_UPLOAD_MIME_TYPES),
                },
            )

        payload, content_hash = self._read_and_hash(stream)

        existing = await self.session.scalar(
            select(Document).where(
                Document.project_id == project_id,
                Document.content_hash == content_hash,
            )
        )
        if existing is not None:
            raise ConflictError(
                "This document has already been ingested into the project.",
                details={"document_id": str(existing.id), "content_hash": content_hash},
            )

        document_id = uuid4()
        storage_key = f"projects/{project_id}/documents/{document_id}"
        await self.storage.put(storage_key, io.BytesIO(payload), content_type=mime_type)

        document = Document(
            id=document_id,
            title=filename or str(document_id),
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(payload),
            content_hash=content_hash,
            storage_key=storage_key,
            ingestion_status=IngestionStatus.PENDING,
            project_id=project_id,
            organization_id=principal.organization_id,
            user_id=principal.user_id,
            metadata_={},
        )
        self.session.add(document)

        # Commit before the job is enqueued: the worker may pick it up
        # immediately and must not race a transaction that has not landed.
        await self.session.commit()
        return document

    def _read_and_hash(self, stream: BinaryIO) -> tuple[bytes, str]:
        """Read the upload, enforcing the size limit as it goes.

        The limit is checked while streaming rather than after, so an oversized
        upload cannot exhaust memory first.
        """
        digest = hashlib.sha256()
        buffer = bytearray()
        while block := stream.read(_READ_BLOCK_BYTES):
            buffer.extend(block)
            digest.update(block)
            if len(buffer) > self.settings.MAX_UPLOAD_BYTES:
                raise PayloadTooLargeError(
                    "The file exceeds the maximum upload size.",
                    details={"max_bytes": self.settings.MAX_UPLOAD_BYTES},
                )
        if not buffer:
            raise UnsupportedMediaTypeError("The uploaded file is empty.")
        return bytes(buffer), digest.hexdigest()

    # --- pipeline --------------------------------------------------------

    async def run_ingestion(self, document_id: UUID) -> Document:
        """Parse and chunk a stored document.

        Called by the worker. An expected failure is recorded on the document
        with its error code rather than raised, so a malformed file does not
        retry forever.
        """
        document = await self.session.get(Document, document_id)
        if document is None:
            raise NotFoundError("Document not found.", details={"document_id": str(document_id)})

        try:
            document.ingestion_status = IngestionStatus.PARSING
            await self.session.commit()

            payload = await self.storage.get(document.storage_key)
            parsed = get_parser(document.mime_type).parse(payload, filename=document.filename)

            document.ingestion_status = IngestionStatus.CHUNKING
            await self.session.commit()

            chunker = RecursiveCharacterChunker(
                chunk_size=self.settings.CHUNK_SIZE_CHARS,
                overlap=self.settings.CHUNK_OVERLAP_CHARS,
            )
            chunks = chunker.chunk(parsed)

            # Clear any previous chunks first: arq retries failed jobs, and a
            # document may be re-ingested after a parser improvement. Without
            # this, a re-run violates the (document_id, ordinal) constraint
            # instead of simply producing the same result.
            await self.session.execute(delete(Chunk).where(Chunk.document_id == document.id))

            self.session.add_all(
                [
                    Chunk(
                        document_id=document.id,
                        ordinal=chunk.ordinal,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        page=chunk.page,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        project_id=document.project_id,
                        organization_id=document.organization_id,
                        user_id=document.user_id,
                        metadata_={},
                    )
                    for chunk in chunks
                ]
            )

            document.page_count = parsed.page_count
            document.chunk_count = len(chunks)
            document.ingestion_status = IngestionStatus.COMPLETED
            document.error_code = None
            document.error_message = None
            await self.session.commit()

            logger.info(
                "ingestion.completed",
                document_id=str(document.id),
                chunk_count=len(chunks),
                page_count=parsed.page_count,
            )
        except DomainError as exc:
            await self.session.rollback()
            await self._record_failure(document_id, exc.code, exc.message)
            logger.warning(
                "ingestion.failed",
                document_id=str(document_id),
                code=str(exc.code),
            )
        except Exception:
            await self.session.rollback()
            await self._record_failure(
                document_id,
                ErrorCode.INGESTION_FAILED,
                "Ingestion failed unexpectedly.",
            )
            logger.exception("ingestion.crashed", document_id=str(document_id))
            raise

        refreshed = await self.session.get(Document, document_id)
        if refreshed is None:  # pragma: no cover - defensive
            raise NotFoundError(
                "Document disappeared during ingestion.",
                details={"document_id": str(document_id)},
            )
        return refreshed

    async def _record_failure(self, document_id: UUID, code: ErrorCode, message: str) -> None:
        document = await self.session.get(Document, document_id)
        if document is None:  # pragma: no cover - defensive
            return
        document.ingestion_status = IngestionStatus.FAILED
        document.error_code = str(code)
        document.error_message = message
        await self.session.commit()

    # --- reads -----------------------------------------------------------

    async def list_documents(self, project_id: UUID) -> list[Document]:
        result = await self.session.scalars(
            select(Document)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at.desc())
        )
        return list(result)

    async def get_document(self, document_id: UUID, project_id: UUID) -> Document:
        document = await self.session.scalar(
            select(Document).where(Document.id == document_id, Document.project_id == project_id)
        )
        if document is None:
            raise NotFoundError("Document not found.", details={"document_id": str(document_id)})
        return document

    async def list_chunks(self, document_id: UUID, project_id: UUID) -> list[Chunk]:
        await self.get_document(document_id, project_id)
        result = await self.session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
        )
        return list(result)
