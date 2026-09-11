"""Ingested documents and their processing state."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: source_id, title, uri, content_hash, mime_type, page_count,
    ingestion_status, error_code, metadata_, storage_key."""

    __tablename__ = "documents"

