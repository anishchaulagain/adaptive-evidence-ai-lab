"""Documents — upload and inspect (spec sections 10, 44)."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import PrincipalDep, ProjectScopeDep, SessionDep, SettingsDep
from app.schemas.document import ChunkRead, DocumentRead
from app.services.ingestion_service import IngestionService
from app.workers.queue import enqueue

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    project_id: ProjectScopeDep,
    file: Annotated[UploadFile, File()],
) -> DocumentRead:
    """Store the file and queue its ingestion.

    Returns 202: parsing and chunking run in a worker, so the response says
    the document was accepted, not that it is ready. Poll the document until
    `ingestion_status` reaches `completed` or `failed`.
    """
    service = IngestionService(session, settings)
    document = await service.create_from_upload(
        stream=file.file,
        filename=file.filename,
        declared_mime_type=file.content_type,
        project_id=project_id,
        principal=principal,
    )
    await enqueue("ingest_document", document.id)
    return DocumentRead.model_validate(document)


@router.get("", response_model=list[DocumentRead])
async def list_documents(
    session: SessionDep, settings: SettingsDep, project_id: ProjectScopeDep
) -> list[DocumentRead]:
    documents = await IngestionService(session, settings).list_documents(project_id)
    return [DocumentRead.model_validate(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID,
    session: SessionDep,
    settings: SettingsDep,
    project_id: ProjectScopeDep,
) -> DocumentRead:
    document = await IngestionService(session, settings).get_document(document_id, project_id)
    return DocumentRead.model_validate(document)


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def list_document_chunks(
    document_id: UUID,
    session: SessionDep,
    settings: SettingsDep,
    project_id: ProjectScopeDep,
) -> list[ChunkRead]:
    """The chunks a document produced, in order, with their provenance."""
    chunks = await IngestionService(session, settings).list_chunks(document_id, project_id)
    return [ChunkRead.model_validate(chunk) for chunk in chunks]
