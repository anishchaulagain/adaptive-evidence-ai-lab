"""Chunks — the retrieval unit, carrying provenance back to the document."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin

# Width of the stored vectors, fixed by mistral-embed. The schema owns this
# number: changing embedding model means altering the column and rebuilding
# the index, so `validate_runtime_settings` asserts settings agree with it.
EMBEDDING_DIMENSIONS = 1024


class Chunk(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """A span of a document.

    `page`, `char_start` and `char_end` exist so every citation resolves to an
    exact span in the source (spec principle 3 — provenance). Slicing the
    document text by those offsets returns this chunk's text.

    `embedding` is null until the embedding worker has run, so a chunk is
    retrievable by keyword before it is retrievable by vector. The full-text
    column arrives in Phase 4.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_id_ordinal"),
        # HNSW with cosine distance. Mistral embeddings are unit-norm, so
        # cosine and inner product rank identically; cosine is used because it
        # stays correct if a future model emits un-normalised vectors.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
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

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", nullable=False, default=dict)
