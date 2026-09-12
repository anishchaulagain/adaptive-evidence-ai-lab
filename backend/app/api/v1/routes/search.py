"""Retrieval endpoint (spec sections 68-69, Phases 3-4).

Deliberately separate from `/query`: this returns evidence and scores only, no
generated answer. `/query` (Phase 7) composes retrieval with generation and
verification on top of exactly this surface, and keeping them apart means
retrieval can be evaluated without an LLM in the loop.

Every hit reports which retriever produced it and at what score, so strategies
can be compared rather than assumed (spec section 15).
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.api.deps import EmbeddingsDep, PrincipalDep, SessionDep
from app.retrieval.pgvector import PgVectorRetriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from app.schemas.search import (
    EvidenceItem,
    SearchRequest,
    SearchResponse,
    SearchStrategy,
)
from app.services.project_service import ProjectService
from core.errors import ErrorCode, ProviderError
from core.retrieval.base import RetrievalQuery
from core.types import RetrievedChunk

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    principal: PrincipalDep,
    session: SessionDep,
    embeddings: EmbeddingsDep,
) -> SearchResponse:
    """Retrieve evidence from a project's chunks using the chosen strategy."""
    # Enforces project isolation before anything is embedded or queried.
    project = await ProjectService(session).get(payload.project_id, principal)

    query = RetrievalQuery(text=payload.query, project_id=project.id, top_k=payload.top_k)

    started = time.perf_counter()
    hits: list[RetrievedChunk]
    embedding_model: str | None = None
    query_terms: list[str] = []

    if payload.strategy is SearchStrategy.KEYWORD:
        keyword = PostgresFtsRetriever(session)
        hits = await keyword.retrieve(query)
        query_terms = await keyword.query_lexemes(payload.query)
    else:
        if embeddings is None:
            raise ProviderError(
                "MISTRAL_API_KEY is not set, so semantic search is unavailable. "
                "Keyword search needs no provider and still works.",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
                provider="mistral",
            )
        # Embedding is the slow, billable part of a semantic search, so it sits
        # inside the timing window rather than hidden from it.
        vector = await embeddings.embed_query(payload.query)
        embedding_model = embeddings.model_id
        hits = await PgVectorRetriever(session).retrieve_with_vector(query, vector)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    return SearchResponse(
        query=payload.query,
        strategy=payload.strategy,
        embedding_model=embedding_model,
        query_terms=query_terms,
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
                matched_terms=list(hit.matched_terms),
            )
            for hit in hits
        ],
    )
