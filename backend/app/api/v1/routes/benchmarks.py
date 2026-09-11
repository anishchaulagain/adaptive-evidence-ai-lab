"""AE-Bench (spec sections 29, 30, 44)."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.benchmark import BenchmarkRunCreate, BenchmarkRunRead

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkRunRead])
async def list_benchmarks(
    principal: PrincipalDep, session: SessionDep
) -> list[BenchmarkRunRead]:
    raise NotImplementedError


@router.post("/run", response_model=BenchmarkRunRead, status_code=status.HTTP_202_ACCEPTED)
async def run_benchmark(
    payload: BenchmarkRunCreate, principal: PrincipalDep, session: SessionDep
) -> BenchmarkRunRead:
    """Enqueue a benchmark run. Seed and code version are recorded so results
    are reproducible — never fabricate or extrapolate them (spec rule 10)."""
    raise NotImplementedError
