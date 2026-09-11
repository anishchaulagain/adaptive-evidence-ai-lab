"""Development-mode data bootstrap.

While `AUTH_MODE=disabled` (Phase 1) there is no registration flow, but
tenancy columns are `NOT NULL` foreign keys. Seeding a fixed organization and
user keeps referential integrity real rather than deferring it, so the
constraints are exercised from the first phase.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.core.security import (
    DEV_ORGANIZATION_ID,
    DEV_USER_EMAIL,
    DEV_USER_ID,
)
from app.models.user import Organization, User

logger = get_logger(__name__)


async def ensure_dev_principal(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    """Idempotently create the development organization and user."""
    async with sessionmaker() as session:
        organization = await session.scalar(
            select(Organization).where(Organization.id == DEV_ORGANIZATION_ID)
        )
        if organization is None:
            session.add(
                Organization(id=DEV_ORGANIZATION_ID, name="Development", slug="development")
            )

        user = await session.scalar(select(User).where(User.id == DEV_USER_ID))
        if user is None:
            session.add(
                User(
                    id=DEV_USER_ID,
                    email=DEV_USER_EMAIL,
                    full_name="Development User",
                    organization_id=DEV_ORGANIZATION_ID,
                    is_active=True,
                )
            )

        if session.new:
            await session.commit()
            logger.info("bootstrap.dev_principal_created")
