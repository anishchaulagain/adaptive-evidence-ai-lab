"""Chunks — the retrieval unit, carrying provenance back to the document."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: document_id, ordinal, text, token_count, embedding (pgvector),
    tsv (full-text), page, char_start, char_end, section, metadata_.

    `page`/`char_start`/`char_end` exist so every citation is verifiable
    (spec principle 3 — provenance).
    """

    __tablename__ = "chunks"

