"""Ingestion vertical slice: upload -> storage -> worker -> chunks.

The worker job is invoked directly rather than through a running arq process.
That keeps the test deterministic while still exercising the real pipeline,
the real object storage and the real database.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sqlalchemy import text

from app.core.config import Settings
from app.db.session import get_sessionmaker
from app.models.document import IngestionStatus
from app.services.ingestion_service import IngestionService
from app.workers.tasks.ingestion import ingest_document

pytestmark = pytest.mark.integration

UPLOAD = "/api/v1/documents/upload"


def _pdf(pages: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for page_text in pages:
        pdf.drawString(72, 720, page_text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
async def project_id(client: AsyncClient) -> AsyncIterator[UUID]:
    """A project to ingest into, removed afterwards.

    Deleting the project cascades to its documents and chunks, which is itself
    a check that the foreign keys are wired correctly.
    """
    response = await client.post("/api/v1/projects", json={"name": f"Ingestion {uuid4().hex[:8]}"})
    identifier = UUID(response.json()["id"])
    yield identifier

    await _delete_project(identifier)


async def _delete_project(identifier: UUID) -> None:
    """Remove a project and, by cascade, its documents and chunks."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": str(identifier)})
        await session.commit()


async def _run_worker(settings: Settings, document_id: UUID) -> None:
    """Invoke the ingestion job exactly as the arq worker would."""
    await ingest_document({"settings": settings}, document_id)


async def _upload(
    client: AsyncClient,
    project_id: UUID,
    *,
    content: bytes,
    filename: str,
    content_type: str,
) -> dict[str, object]:
    response = await client.post(
        UPLOAD,
        params={"project_id": str(project_id)},
        files={"file": (filename, content, content_type)},
    )
    assert response.status_code == 202, response.text
    body: dict[str, object] = response.json()
    return body


# --- upload ---------------------------------------------------------------


async def test_upload_accepts_and_queues_without_blocking(
    client: AsyncClient, project_id: UUID
) -> None:
    """202, not 200: the document is accepted, not yet ingested."""
    body = await _upload(
        client,
        project_id,
        content=b"# Title\n\nSome body text.",
        filename="notes.md",
        content_type="text/markdown",
    )

    assert body["ingestion_status"] == IngestionStatus.PENDING
    assert body["chunk_count"] == 0
    assert body["mime_type"] == "text/markdown"
    assert body["size_bytes"] == len(b"# Title\n\nSome body text.")
    assert body["content_hash"]


async def test_mime_type_is_resolved_from_the_extension(
    client: AsyncClient, project_id: UUID
) -> None:
    """Clients routinely send octet-stream; a known extension must win."""
    body = await _upload(
        client,
        project_id,
        content=b"plain text body",
        filename="notes.txt",
        content_type="application/octet-stream",
    )

    assert body["mime_type"] == "text/plain"


async def test_unsupported_type_is_rejected(client: AsyncClient, project_id: UUID) -> None:
    response = await client.post(
        UPLOAD,
        params={"project_id": str(project_id)},
        files={"file": ("sheet.xlsx", b"binary", "application/vnd.ms-excel")},
    )

    assert response.status_code == 415
    error = response.json()["error"]
    assert error["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert "application/pdf" in error["details"]["supported"]


async def test_empty_file_is_rejected(client: AsyncClient, project_id: UUID) -> None:
    response = await client.post(
        UPLOAD,
        params={"project_id": str(project_id)},
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 415


async def test_oversized_upload_is_rejected(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    oversized = b"x" * (settings.MAX_UPLOAD_BYTES + 1)

    response = await client.post(
        UPLOAD,
        params={"project_id": str(project_id)},
        files={"file": ("big.txt", oversized, "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


async def test_reuploading_the_same_content_conflicts(
    client: AsyncClient, project_id: UUID
) -> None:
    """Deduplicated by content hash, so the same file is not embedded twice.
    The response names the existing document so a client can recover."""
    first = await _upload(
        client,
        project_id,
        content=b"identical content",
        filename="a.txt",
        content_type="text/plain",
    )

    response = await client.post(
        UPLOAD,
        params={"project_id": str(project_id)},
        files={"file": ("differently-named.txt", b"identical content", "text/plain")},
    )

    assert response.status_code == 409
    assert response.json()["error"]["details"]["document_id"] == first["id"]


async def test_upload_to_an_unknown_project_is_not_found(client: AsyncClient) -> None:
    response = await client.post(
        UPLOAD,
        params={"project_id": str(uuid4())},
        files={"file": ("a.txt", b"body", "text/plain")},
    )

    assert response.status_code == 404


# --- pipeline -------------------------------------------------------------


async def test_text_document_is_parsed_and_chunked(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    body = await _upload(
        client,
        project_id,
        content=b"First paragraph.\n\nSecond paragraph.",
        filename="notes.txt",
        content_type="text/plain",
    )
    document_id = UUID(str(body["id"]))

    await _run_worker(settings, document_id)

    document = (
        await client.get(f"/api/v1/documents/{document_id}", params={"project_id": str(project_id)})
    ).json()
    assert document["ingestion_status"] == IngestionStatus.COMPLETED
    assert document["chunk_count"] == 2
    assert document["error_code"] is None

    chunks = (
        await client.get(
            f"/api/v1/documents/{document_id}/chunks", params={"project_id": str(project_id)}
        )
    ).json()
    assert [chunk["text"] for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert [chunk["ordinal"] for chunk in chunks] == [0, 1]
    assert all(chunk["token_count"] > 0 for chunk in chunks)


async def test_pdf_chunks_carry_page_numbers(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    """Page provenance is what makes a PDF citation verifiable."""
    body = await _upload(
        client,
        project_id,
        content=_pdf(["Alpha page content", "Beta page content"]),
        filename="report.pdf",
        content_type="application/pdf",
    )
    document_id = UUID(str(body["id"]))

    await _run_worker(settings, document_id)

    document = (
        await client.get(f"/api/v1/documents/{document_id}", params={"project_id": str(project_id)})
    ).json()
    assert document["ingestion_status"] == IngestionStatus.COMPLETED
    assert document["page_count"] == 2

    chunks = (
        await client.get(
            f"/api/v1/documents/{document_id}/chunks", params={"project_id": str(project_id)}
        )
    ).json()
    assert sorted({chunk["page"] for chunk in chunks}) == [1, 2]
    assert all(chunk["char_start"] is not None for chunk in chunks)


async def test_a_corrupt_pdf_fails_with_an_attributable_code(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    """The failure must name its stage rather than surfacing as a bare 500."""
    body = await _upload(
        client,
        project_id,
        content=b"%PDF-1.4 this is not actually a pdf",
        filename="broken.pdf",
        content_type="application/pdf",
    )
    document_id = UUID(str(body["id"]))

    await _run_worker(settings, document_id)

    document = (
        await client.get(f"/api/v1/documents/{document_id}", params={"project_id": str(project_id)})
    ).json()
    assert document["ingestion_status"] == IngestionStatus.FAILED
    assert document["error_code"] == "PARSING_FAILED"
    assert document["error_message"]
    assert document["chunk_count"] == 0


async def test_re_running_ingestion_is_idempotent(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    """arq retries failed jobs, so a second run must converge on the same
    result rather than duplicating chunks or tripping a constraint."""
    body = await _upload(
        client,
        project_id,
        content=b"First paragraph.\n\nSecond paragraph.",
        filename="notes.txt",
        content_type="text/plain",
    )
    document_id = UUID(str(body["id"]))

    await _run_worker(settings, document_id)
    first = (
        await client.get(
            f"/api/v1/documents/{document_id}/chunks", params={"project_id": str(project_id)}
        )
    ).json()

    await _run_worker(settings, document_id)
    second = (
        await client.get(
            f"/api/v1/documents/{document_id}/chunks", params={"project_id": str(project_id)}
        )
    ).json()

    assert [c["text"] for c in second] == [c["text"] for c in first]
    assert [c["ordinal"] for c in second] == [0, 1]

    document = (
        await client.get(f"/api/v1/documents/{document_id}", params={"project_id": str(project_id)})
    ).json()
    assert document["chunk_count"] == 2


# --- reads ----------------------------------------------------------------


async def test_documents_are_scoped_to_their_project(client: AsyncClient, project_id: UUID) -> None:
    body = await _upload(
        client,
        project_id,
        content=b"scoped content",
        filename="a.txt",
        content_type="text/plain",
    )
    other = await client.post("/api/v1/projects", json={"name": f"Other {uuid4().hex[:8]}"})
    other_id = other.json()["id"]
    try:
        response = await client.get(
            f"/api/v1/documents/{body['id']}", params={"project_id": other_id}
        )

        assert response.status_code == 404
    finally:
        await _delete_project(UUID(other_id))


async def test_listing_returns_the_projects_documents(
    client: AsyncClient, project_id: UUID
) -> None:
    await _upload(client, project_id, content=b"one", filename="a.txt", content_type="text/plain")
    await _upload(client, project_id, content=b"two", filename="b.txt", content_type="text/plain")

    response = await client.get("/api/v1/documents", params={"project_id": str(project_id)})

    assert response.status_code == 200
    assert {item["filename"] for item in response.json()} == {"a.txt", "b.txt"}


async def test_chunks_for_an_unknown_document_are_not_found(
    client: AsyncClient, project_id: UUID
) -> None:
    response = await client.get(
        f"/api/v1/documents/{uuid4()}/chunks", params={"project_id": str(project_id)}
    )

    assert response.status_code == 404


async def test_storage_holds_the_uploaded_bytes(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    """The stored object must be byte-identical, or re-parsing later is unsound."""
    content = b"exact bytes to store"
    body = await _upload(
        client, project_id, content=content, filename="a.txt", content_type="text/plain"
    )

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        service = IngestionService(session, settings)
        document = await service.get_document(UUID(str(body["id"])), project_id)
        stored = await service.storage.get(document.storage_key)

    assert stored == content


async def test_a_job_for_a_deleted_document_is_terminal(
    client: AsyncClient, project_id: UUID, settings: Settings
) -> None:
    """Deleting a project while its job is queued is a normal race. The job
    must not raise, or arq would retry something that can never succeed."""
    body = await _upload(
        client,
        project_id,
        content=b"about to be deleted",
        filename="doomed.txt",
        content_type="text/plain",
    )
    document_id = UUID(str(body["id"]))
    await _delete_project(project_id)

    await _run_worker(settings, document_id)
