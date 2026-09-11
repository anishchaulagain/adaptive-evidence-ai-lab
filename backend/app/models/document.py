"""Ingested documents and their processing state (spec section 10)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from sqlalchemy import Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class IngestionStatus(StrEnum):
    """Lifecycle of a document through the pipeline.

    Stored as text rather than a native enum: later phases add states, and a
    string column avoids a migration-time type alteration each time.
    """

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """A file that has been accepted for ingestion.

    `content_hash` is unique per project so the same file is not ingested and
    embedded twice. `error_code` holds an `ErrorCode` value when ingestion
    fails, making the failure attributable (spec section 47).
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("project_id", "content_hash", name="uq_documents_project_id_content_hash"),
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(150), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    ingestion_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=IngestionStatus.PENDING, index=True
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", nullable=False, default=dict)
