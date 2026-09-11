"""Document schemas."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class DocumentRead(IdentifiedModel):
    """Fields: source_id, title, mime_type, page_count, ingestion_status,
    error_code, chunk_count."""


class ChunkRead(APIModel):
    """Fields: id, document_id, ordinal, text, page, char_start, char_end.

    Provenance fields are mandatory so a citation can be resolved back to the
    exact span (spec principle 3).
    """
