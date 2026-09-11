"""Experiment lab (spec sections 31-34, 44)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.experiment import ExperimentCreate, ExperimentRead

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreate, principal: PrincipalDep, session: SessionDep
) -> ExperimentRead:
    """Create an experiment. The configuration is frozen so the run is
    reproducible (spec principle 2)."""
    raise NotImplementedError


@router.get("", response_model=list[ExperimentRead])
async def list_experiments(principal: PrincipalDep, session: SessionDep) -> list[ExperimentRead]:
    raise NotImplementedError


@router.post(
    "/{experiment_id}/run",
    response_model=ExperimentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_experiment(
    experiment_id: UUID, principal: PrincipalDep, session: SessionDep
) -> ExperimentRead:
    raise NotImplementedError
