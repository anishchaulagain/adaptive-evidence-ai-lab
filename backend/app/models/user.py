"""Identity: users and organizations (spec section 43)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Organization(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A tenant boundary. Columns: name, slug, settings."""

    __tablename__ = "organizations"


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Columns: email, hashed_password, organization_id, role, is_active."""

    __tablename__ = "users"

