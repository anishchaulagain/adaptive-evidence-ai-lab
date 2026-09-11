"""Documents — upload and inspect (spec sections 10, 44)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.document import DocumentRead

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    principal: PrincipalDep,
    session: SessionDep,
    file: UploadFile = File(...),
) -> DocumentRead:
    """Persist the file to object storage and enqueue ingestion.

    Returns 202: parsing, chunking and embedding happen in a worker.
    """
    raise NotImplementedError


@router.get("", response_model=list[DocumentRead])
async def list_documents(principal: PrincipalDep, session: SessionDep) -> list[DocumentRead]:
    raise NotImplementedError


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID, principal: PrincipalDep, session: SessionDep
) -> DocumentRead:
    raise NotImplementedError
