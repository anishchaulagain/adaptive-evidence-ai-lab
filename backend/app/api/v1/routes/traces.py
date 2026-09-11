"""Trace retrieval — every AI operation is inspectable (spec section 23)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.trace import TraceRead

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("/{trace_id}", response_model=TraceRead)
async def get_trace(
    trace_id: UUID, principal: PrincipalDep, session: SessionDep
) -> TraceRead:
    """Return the full execution trace: stages, timings, tokens, cost, errors."""
    raise NotImplementedError
