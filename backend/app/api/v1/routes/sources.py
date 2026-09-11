"""Data sources — connect, list and disconnect (spec sections 9, 44)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.source import SourceCreate, SourceRead

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post("", response_model=SourceRead, status_code=status.HTTP_201_CREATED)
async def create_source(
    payload: SourceCreate, principal: PrincipalDep, session: SessionDep
) -> SourceRead:
    """Register a source. Ingestion is dispatched to a worker, not run inline
    (spec section 46)."""
    raise NotImplementedError


@router.get("", response_model=list[SourceRead])
async def list_sources(principal: PrincipalDep, session: SessionDep) -> list[SourceRead]:
    raise NotImplementedError


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: UUID, principal: PrincipalDep, session: SessionDep
) -> None:
    raise NotImplementedError
