"""Retrieval endpoint (spec section 68, Phase 3).

Deliberately separate from `/query`: this returns evidence and scores only, no
generated answer. `/query` (Phase 7) composes retrieval with generation and
verification on top of exactly this surface, and keeping them apart means
retrieval can be evaluated without an LLM in the loop.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.api.deps import EmbeddingsDep, PrincipalDep, SessionDep
from app.retrieval.pgvector import PgVectorRetriever
from app.schemas.search import EvidenceItem, SearchRequest, SearchResponse
from app.services.project_service import ProjectService
from core.retrieval.base import RetrievalQuery

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    principal: PrincipalDep,
    session: SessionDep,
    embeddings: EmbeddingsDep,
) -> SearchResponse:
    """Dense retrieval over a project's embedded chunks."""
    # Enforces project isolation before anything is embedded or queried.
    project = await ProjectService(session).get(payload.project_id, principal)

    started = time.perf_counter()
    vector = await embeddings.embed_query(payload.query)

    retriever = PgVectorRetriever(session)
    hits = await retriever.retrieve_with_vector(
        RetrievalQuery(text=payload.query, project_id=project.id, top_k=payload.top_k),
        vector,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    return SearchResponse(
        query=payload.query,
        strategy="semantic",
        embedding_model=embeddings.model_id,
        top_k=payload.top_k,
        latency_ms=latency_ms,
        results=[
            EvidenceItem(
                chunk_id=hit.provenance.chunk_id,
                document_id=hit.provenance.document_id,
                text=hit.text,
                score=hit.score,
                retriever=str(hit.retriever),
                rank=hit.rank,
                page=hit.provenance.page,
                char_start=hit.provenance.char_start,
                char_end=hit.provenance.char_end,
            )
            for hit in hits
        ],
    )
