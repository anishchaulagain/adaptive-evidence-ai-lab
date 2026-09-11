"""Projects — the unit of isolation for data, runs and experiments."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Project(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: name, description, settings, default_model_config_id."""

    __tablename__ = "projects"

