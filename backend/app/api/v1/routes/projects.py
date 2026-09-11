"""Projects — create and list (spec section 44)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.project import ProjectCreate, ProjectRead

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: ProjectCreate, principal: PrincipalDep, session: SessionDep
) -> ProjectRead:
    raise NotImplementedError


@router.get("", response_model=list[ProjectRead])
async def list_projects(principal: PrincipalDep, session: SessionDep) -> list[ProjectRead]:
    raise NotImplementedError


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(
    project_id: UUID, principal: PrincipalDep, session: SessionDep
) -> ProjectRead:
    raise NotImplementedError
