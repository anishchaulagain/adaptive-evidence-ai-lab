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

from app.core.config import AuthMode, Settings, get_settings
from app.core.errors import UnauthenticatedError
from app.core.providers import get_embedding_pipeline
from app.core.security import DEV_PRINCIPAL, Principal, decode_access_token
from app.db.session import get_sessionmaker
from core.embeddings.pipeline import BatchedEmbeddingPipeline


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
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the authenticated caller.

    Phase 1 runs with `AUTH_MODE=disabled` and returns a fixed development
    principal, seeded at startup. `validate_runtime_settings()` refuses to boot
    production in that mode, so this cannot ship by accident.
    """
    if settings.AUTH_MODE is AuthMode.DISABLED:
        return DEV_PRINCIPAL

    if authorization is None or not authorization.lower().startswith("bearer "):
        raise UnauthenticatedError("Missing or malformed Authorization header.")
    return decode_access_token(authorization.split(" ", 1)[1])


async def get_embeddings(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[BatchedEmbeddingPipeline]:
    """Yield the configured embedding pipeline, closing its connection pool.

    Exposed as a dependency rather than constructed in the route so tests can
    substitute a deterministic embedder and run without a provider key.
    """
    pipeline = get_embedding_pipeline(settings)
    try:
        yield pipeline
    finally:
        await pipeline.aclose()


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingsDep = Annotated[BatchedEmbeddingPipeline, Depends(get_embeddings)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
PrincipalDep = Annotated[Principal, Depends(get_current_principal)]


async def get_project_scope(
    project_id: UUID,
    principal: PrincipalDep,
    session: SessionDep,
) -> UUID:
    """Verify the principal may access `project_id`.

    Spec section 42: project isolation is enforced here, not in each route.
    Raises `NotFoundError` — never `ForbiddenError` — for a project in another
    organization, so the API does not leak which project IDs exist.
    """
    from app.services.project_service import ProjectService

    project = await ProjectService(session).get(project_id, principal)
    return project.id


ProjectScopeDep = Annotated[UUID, Depends(get_project_scope)]

__all__ = [
    "EmbeddingsDep",
    "PrincipalDep",
    "ProjectScopeDep",
    "SessionDep",
    "SettingsDep",
    "get_current_principal",
    "get_db_session",
    "get_embeddings",
    "get_project_scope",
]
