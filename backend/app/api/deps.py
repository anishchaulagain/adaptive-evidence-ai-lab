"""Shared FastAPI dependencies.

Keep dependencies thin: resolve identity, open a session, scope to a project.
Business logic belongs in `app.services`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.context import bind_context
from app.core.security import Principal
from app.db.session import get_sessionmaker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a transactional session, committing on success."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_current_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the authenticated caller from the Authorization header."""
    raise NotImplementedError


async def get_project_scope(
    project_id: UUID,
    principal: Annotated[Principal, Depends(get_current_principal)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UUID:
    """Verify the principal may access `project_id` and bind it to the context.

    Spec section 42: project isolation is enforced here, not in each route.
    """
    raise NotImplementedError


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
PrincipalDep = Annotated[Principal, Depends(get_current_principal)]
ProjectScopeDep = Annotated[UUID, Depends(get_project_scope)]

__all__ = [
    "PrincipalDep",
    "ProjectScopeDep",
    "SessionDep",
    "SettingsDep",
    "bind_context",
    "get_current_principal",
    "get_db_session",
    "get_project_scope",
]
