"""Project lifecycle and tenant-scoped access (spec sections 42, 43)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.errors import ConflictError, NotFoundError
from app.core.security import Principal
from app.models.project import Project
from app.schemas.project import ProjectCreate
from app.services.base import Service


class ProjectService(Service):
    """Every query filters on the principal's organization, so isolation holds
    even if a caller guesses a valid project ID."""

    async def create(self, payload: ProjectCreate, principal: Principal) -> Project:
        project = Project(
            name=payload.name,
            description=payload.description,
            settings=payload.settings,
            organization_id=principal.organization_id,
            user_id=principal.user_id,
        )
        self.session.add(project)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError(
                f"A project named {payload.name!r} already exists.",
                details={"name": payload.name},
            ) from exc
        return project

    async def list(self, principal: Principal) -> list[Project]:
        result = await self.session.scalars(
            select(Project)
            .where(Project.organization_id == principal.organization_id)
            .order_by(Project.created_at.desc())
        )
        return list(result)

    async def get(self, project_id: UUID, principal: Principal) -> Project:
        """Fetch a project the principal owns.

        Raises `NotFoundError` for a project in another organization rather
        than `ForbiddenError`, so the API does not disclose its existence.
        """
        project = await self.session.scalar(
            select(Project).where(
                Project.id == project_id,
                Project.organization_id == principal.organization_id,
            )
        )
        if project is None:
            raise NotFoundError("Project not found.", details={"project_id": str(project_id)})
        return project
