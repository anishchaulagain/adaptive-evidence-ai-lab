"""User queries and their generated answers."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Query(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: text, analysis (difficulty/intent), retrieval_config,
    answer, citations, trace_id, status, error_code."""

    __tablename__ = "queries"

