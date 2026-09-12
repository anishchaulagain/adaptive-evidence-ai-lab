"""Query execution (spec sections 13, 22, 44, 72).

Composes retrieval with generation: retrieve evidence, answer strictly from it,
resolve every citation back to an exact span. `/search` remains the retrieval
surface on its own, so retrieval can still be evaluated without an LLM.

Streaming (spec section 45) and the trace system (section 23) arrive in the
phases that own them; this endpoint returns the complete answer.
"""

from __future__ import annotations

import time
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import EmbeddingsDep, GeneratorDep, PrincipalDep, SessionDep, SettingsDep
from app.api.v1.routes.search import to_evidence
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from app.schemas.query import (
    CitationRef,
    ClaimRead,
    QueryRequest,
    QueryResponse,
    QueryUsage,
)
from app.schemas.search import SearchStrategy
from app.services.project_service import ProjectService
from core.errors import ErrorCode, ProviderError
from core.reasoning.base import InferenceBudget
from core.retrieval.base import RetrievalQuery
from core.types import RetrievedChunk

router = APIRouter(prefix="/query", tags=["query"])

_NEEDS_EMBEDDINGS = frozenset({SearchStrategy.SEMANTIC, SearchStrategy.HYBRID})


def _citation(chunk: RetrievedChunk) -> CitationRef:
    return CitationRef(
        chunk_id=chunk.provenance.chunk_id,
        document_id=chunk.provenance.document_id,
        page=chunk.provenance.page,
        char_start=chunk.provenance.char_start,
        char_end=chunk.provenance.char_end,
        text=chunk.text,
    )


@router.post("", response_model=QueryResponse)
async def create_query(
    payload: QueryRequest,
    principal: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    embeddings: EmbeddingsDep,
    generator: GeneratorDep,
) -> QueryResponse:
    """Answer a question from a project's evidence, with claim-level citations."""
    project = await ProjectService(session).get(payload.project_id, principal)

    if generator is None:
        raise ProviderError(
            "MISTRAL_API_KEY is not set, so answers cannot be generated. "
            "Use /search to retrieve evidence without a provider.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
        )
    if payload.strategy in _NEEDS_EMBEDDINGS and embeddings is None:
        raise ProviderError(
            f"MISTRAL_API_KEY is not set, so {payload.strategy} retrieval is unavailable.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
        )

    query = RetrievalQuery(text=payload.query, project_id=project.id, top_k=payload.top_k)
    keyword = PostgresFtsRetriever(session)
    dense = PgVectorRetriever(session, embeddings)

    retrieval_started = time.perf_counter()
    evidence: list[RetrievedChunk]
    match payload.strategy:
        case SearchStrategy.KEYWORD:
            evidence = await keyword.retrieve(query)
        case SearchStrategy.SEMANTIC:
            evidence = await dense.retrieve(query)
        case SearchStrategy.HYBRID:
            evidence = await HybridRetriever(dense, keyword).retrieve(query)
    retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)

    generation_started = time.perf_counter()
    answer = await generator.generate(
        payload.query,
        evidence,
        InferenceBudget(
            max_input_tokens=0,
            max_output_tokens=settings.GENERATION_MAX_OUTPUT_TOKENS,
        ),
    )
    generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)

    # Resolve cited chunk IDs back to spans. Built from the evidence actually
    # retrieved, so a citation can only ever point at real source text.
    by_id: dict[UUID, RetrievedChunk] = {chunk.provenance.chunk_id: chunk for chunk in evidence}
    claims = [
        ClaimRead(
            text=claim.text,
            evidence=[
                _citation(by_id[chunk_id]) for chunk_id in claim.evidence if chunk_id in by_id
            ],
            supported=claim.is_supported,
        )
        for claim in answer.claims
    ]

    return QueryResponse(
        query=payload.query,
        answer=answer.text,
        abstained=answer.abstained,
        claims=claims,
        strategy=payload.strategy,
        model=answer.model_id,
        embedding_model=(
            embeddings.model_id
            if embeddings is not None and payload.strategy in _NEEDS_EMBEDDINGS
            else None
        ),
        evidence=[to_evidence(hit) for hit in evidence] if payload.include_evidence else [],
        evidence_count=len(evidence),
        invented_citations=int(answer.metadata.get("invented_citations", 0)),
        unsupported_claims=len(answer.unsupported_claims),
        usage=QueryUsage(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=round(retrieval_ms + generation_ms, 2),
        ),
    )


@router.post("/stream")
async def stream_query(
    payload: QueryRequest, principal: PrincipalDep, session: SessionDep
) -> StreamingResponse:
    """Stream execution progress and answer tokens as Server-Sent Events.

    Unimplemented: streaming is spec section 45, and arrives with the trace
    system whose stage events it carries.
    """
    raise NotImplementedError
