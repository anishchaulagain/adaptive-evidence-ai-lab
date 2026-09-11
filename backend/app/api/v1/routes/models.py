"""Model registry and configuration (spec sections 7, 8, 44)."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.model import ModelConfigCreate, ModelConfigRead

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelConfigRead])
async def list_models(principal: PrincipalDep, session: SessionDep) -> list[ModelConfigRead]:
    """List configured models. API keys are never included in the response."""
    raise NotImplementedError


@router.post("", response_model=ModelConfigRead, status_code=status.HTTP_201_CREATED)
async def create_model(
    payload: ModelConfigCreate, principal: PrincipalDep, session: SessionDep
) -> ModelConfigRead:
    raise NotImplementedError
