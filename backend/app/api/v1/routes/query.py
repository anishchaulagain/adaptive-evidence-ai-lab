"""Query execution — buffered and streamed (spec sections 13, 44, 45)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import PrincipalDep, SessionDep
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def create_query(
    payload: QueryRequest, principal: PrincipalDep, session: SessionDep
) -> QueryResponse:
    """Run the full pipeline and return the answer, citations and trace ID."""
    raise NotImplementedError


@router.post("/stream")
async def stream_query(
    payload: QueryRequest, principal: PrincipalDep, session: SessionDep
) -> StreamingResponse:
    """Stream execution progress and answer tokens as Server-Sent Events.

    Event sequence mirrors the pipeline stages so the UI can render progress:
    `query_analyzed`, `semantic_retrieval`, `keyword_retrieval`, `fusion`,
    `reranking`, `generating`, `verification`, `done`.
    """
    raise NotImplementedError
