"""Document and chunk schemas.

The `metadata` column is not exposed yet: nothing populates it beyond the
filename, which is already a field of its own. It joins the response when a
connector supplies real source metadata.
"""

from __future__ import annotations

from uuid import UUID

from app.schemas.common import APIModel, IdentifiedModel


class DocumentRead(IdentifiedModel):
    """A document and its ingestion state."""

    project_id: UUID
    title: str
    filename: str | None
    mime_type: str
    size_bytes: int
    content_hash: str
    page_count: int | None
    chunk_count: int
    ingestion_status: str
    error_code: str | None
    error_message: str | None


class ChunkRead(APIModel):
    """A retrieval unit with the provenance needed to verify a citation.

    Slicing the source document by `char_start`/`char_end` returns `text`.
    """

    id: UUID
    document_id: UUID
    ordinal: int
    text: str
    token_count: int
    page: int | None
    char_start: int | None
    char_end: int | None
