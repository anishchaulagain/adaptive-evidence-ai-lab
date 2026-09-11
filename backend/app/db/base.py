"""Declarative base and shared column conventions.

Models themselves live in `app.models`, whose package `__init__` imports every
registered module so that `Base.metadata` is complete for Alembic autogenerate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

# Explicit naming convention keeps migrations deterministic across environments.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: ClassVar[dict[object, object]] = {dict[str, Any]: JSONB}


class UUIDPrimaryKeyMixin:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class OwnedMixin:
    """Ownership columns for top-level entities such as projects.

    Spec section 43 — carry tenancy from day one so multi-tenancy is possible
    later without a schema rewrite.
    """

    @declared_attr
    @classmethod
    def organization_id(cls) -> Mapped[UUID]:
        return mapped_column(
            ForeignKey("organizations.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )

    @declared_attr
    @classmethod
    def user_id(cls) -> Mapped[UUID | None]:
        return mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)


class TenantMixin(OwnedMixin):
    """Ownership plus project scoping, for entities that live inside a project
    (documents, chunks, traces, runs).

    Deleting a project removes everything scoped to it.
    """

    @declared_attr
    @classmethod
    def project_id(cls) -> Mapped[UUID]:
        return mapped_column(
            ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False
        )
