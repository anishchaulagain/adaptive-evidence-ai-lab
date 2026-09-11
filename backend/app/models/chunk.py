"""Chunks — the retrieval unit, carrying provenance back to the document."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """A span of a document.

    `page`, `char_start` and `char_end` exist so every citation resolves to an
    exact span in the source (spec principle 3 — provenance). Slicing the
    document text by those offsets returns this chunk's text.

    The `embedding` (pgvector) and full-text columns arrive in Phases 3 and 4,
    alongside the code that populates them.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_id_ordinal"),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", nullable=False, default=dict)
