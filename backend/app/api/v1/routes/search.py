"""Retrieval endpoint (spec sections 68-70, Phases 3-5).

Deliberately separate from `/query`: this returns evidence and scores only, no
generated answer. `/query` (Phase 7) composes retrieval with generation and
verification on top of exactly this surface, and keeping them apart means
retrieval can be evaluated without an LLM in the loop.

Every hit reports which retriever produced it and at what score, and a fused
hit reports what each arm contributed, so strategies can be compared rather
than assumed (spec sections 15, 17).
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.api.deps import EmbeddingsDep, PrincipalDep, SessionDep
from app.retrieval.adaptive import AdaptiveRetriever
from app.retrieval.factory import EMBEDDING_STRATEGIES, build_retriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from app.schemas.search import (
    EvidenceItem,
    QueryAnalysisRead,
    SearchRequest,
    SearchResponse,
    SearchStrategy,
)
from app.services.project_service import ProjectService
from core.errors import ErrorCode, ProviderError
from core.retrieval.base import RetrievalQuery
from core.types import RetrievedChunk, RetrieverKind

router = APIRouter(prefix="/search", tags=["retrieval"])

_NEEDS_EMBEDDINGS = EMBEDDING_STRATEGIES


def _analysis_read(retriever: AdaptiveRetriever) -> QueryAnalysisRead | None:
    """Flatten the adaptive decision for the response."""
    analysis, profile = retriever.last_analysis, retriever.last_profile
    if analysis is None or profile is None:  # pragma: no cover - set by retrieve()
        return None
    return QueryAnalysisRead(
        query_type=str(analysis.query_type),
        difficulty=analysis.difficulty,
        ambiguity=analysis.ambiguity,
        requires_exact_match=analysis.requires_exact_match,
        requires_multihop=analysis.requires_multihop,
        requires_numeric_reasoning=analysis.requires_numeric_reasoning,
        entities=list(analysis.entities),
        signals=list(analysis.signals),
        semantic_weight=profile.semantic_weight,
        keyword_weight=profile.keyword_weight,
        fetch_multiplier=profile.fetch_multiplier,
        reason=profile.reason,
    )


def to_evidence(hit: RetrievedChunk) -> EvidenceItem:
    """Flatten a hit, including per-retriever contributions (spec section 17)."""
    semantic = hit.contribution(RetrieverKind.SEMANTIC)
    keyword = hit.contribution(RetrieverKind.KEYWORD)
    return EvidenceItem(
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
        retrieval_source=[str(item.retriever) for item in hit.contributions],
        fusion_score=hit.fusion_score,
        semantic_score=semantic.score if semantic else None,
        semantic_rank=semantic.rank if semantic else None,
        keyword_score=keyword.score if keyword else None,
        keyword_rank=keyword.rank if keyword else None,
    )


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

    if str(payload.strategy) in _NEEDS_EMBEDDINGS and embeddings is None:
        raise ProviderError(
            f"MISTRAL_API_KEY is not set, so {payload.strategy} search is "
            "unavailable. Keyword search needs no provider and still works.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
        )

    query = RetrievalQuery(text=payload.query, project_id=project.id, top_k=payload.top_k)
    retriever = build_retriever(str(payload.strategy), session, embeddings)

    # Embedding is the slow, billable part of a search, so it sits inside the
    # timing window rather than hidden from it.
    started = time.perf_counter()
    hits: list[RetrievedChunk] = await retriever.retrieve(query)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    # Lexemes explain a lexical hit, so report them whenever one could occur.
    query_terms: list[str] = []
    if payload.strategy is not SearchStrategy.SEMANTIC:
        query_terms = await PostgresFtsRetriever(session).query_lexemes(payload.query)

    analysis = _analysis_read(retriever) if isinstance(retriever, AdaptiveRetriever) else None

    return SearchResponse(
        query=payload.query,
        strategy=payload.strategy,
        embedding_model=(
            embeddings.model_id
            if embeddings is not None and str(payload.strategy) in _NEEDS_EMBEDDINGS
            else None
        ),
        query_terms=query_terms,
        analysis=analysis,
        top_k=payload.top_k,
        latency_ms=latency_ms,
        results=[to_evidence(hit) for hit in hits],
    )
