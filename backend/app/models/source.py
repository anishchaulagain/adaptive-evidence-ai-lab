"""Data sources connected to a project (spec section 9)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class DataSource(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: name, kind, connection_config, status, last_synced_at, stats."""

    __tablename__ = "data_sources"
