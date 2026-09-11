"""Evaluation lab (spec sections 26-28, 44)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.evaluation import EvaluationRunCreate, EvaluationRunRead

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("", response_model=EvaluationRunRead, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    payload: EvaluationRunCreate, principal: PrincipalDep, session: SessionDep
) -> EvaluationRunRead:
    raise NotImplementedError


@router.get("", response_model=list[EvaluationRunRead])
async def list_evaluations(principal: PrincipalDep, session: SessionDep) -> list[EvaluationRunRead]:
    raise NotImplementedError


@router.post("/run", response_model=EvaluationRunRead, status_code=status.HTTP_202_ACCEPTED)
async def run_evaluation(
    payload: EvaluationRunCreate, principal: PrincipalDep, session: SessionDep
) -> EvaluationRunRead:
    """Enqueue an evaluation run; results stream into the run record."""
    raise NotImplementedError


@router.get("/{run_id}", response_model=EvaluationRunRead)
async def get_evaluation(
    run_id: UUID, principal: PrincipalDep, session: SessionDep
) -> EvaluationRunRead:
    raise NotImplementedError
