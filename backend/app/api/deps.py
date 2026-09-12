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
from app.core.providers import (
    embeddings_enabled,
    generation_enabled,
    get_answer_generator,
    get_embedding_pipeline,
)
from app.core.security import DEV_PRINCIPAL, Principal, decode_access_token
from app.db.session import get_sessionmaker
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from core.reasoning.grounded import GroundedAnswerGenerator


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
) -> AsyncIterator[BatchedEmbeddingPipeline | None]:
    """Yield the configured embedding pipeline, or None when no key is set.

    Yields None rather than raising because dependencies resolve eagerly, and
    keyword retrieval needs no provider at all — it would otherwise be broken
    by a missing key it never uses. The semantic path raises instead.

    Exposed as a dependency so tests can substitute a deterministic embedder.
    """
    if not embeddings_enabled(settings):
        yield None
        return

    pipeline = get_embedding_pipeline(settings)
    try:
        yield pipeline
    finally:
        await pipeline.aclose()


async def get_generator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[GroundedAnswerGenerator | None]:
    """Yield the configured answer generator, or None when no key is set.

    None rather than raising, for the same reason as `get_embeddings`:
    dependencies resolve eagerly, and a route that only retrieves must not be
    broken by a missing key it never uses.
    """
    if not generation_enabled(settings):
        yield None
        return

    generator = get_answer_generator(settings)
    try:
        yield generator
    finally:
        await generator.aclose()


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingsDep = Annotated[BatchedEmbeddingPipeline | None, Depends(get_embeddings)]
GeneratorDep = Annotated[GroundedAnswerGenerator | None, Depends(get_generator)]
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
    "GeneratorDep",
    "PrincipalDep",
    "ProjectScopeDep",
    "SessionDep",
    "SettingsDep",
    "get_current_principal",
    "get_db_session",
    "get_embeddings",
    "get_generator",
    "get_project_scope",
]
