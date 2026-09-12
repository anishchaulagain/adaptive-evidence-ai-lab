"""Query execution (spec sections 13, 22, 23, 44, 72, 73).

Composes retrieval with generation: retrieve evidence, answer strictly from it,
resolve every citation back to an exact span. `/search` remains the retrieval
surface on its own, so retrieval can still be evaluated without an LLM.

Every execution produces a persisted trace, including a failed one — a failure
that leaves no trace is the hardest kind to diagnose (spec section 47).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import (
    EmbeddingsDep,
    GeneratorDep,
    PrincipalDep,
    SessionDep,
    SettingsDep,
)
from app.api.v1.routes.search import to_evidence
from app.core.context import bind_context
from app.core.security import Principal
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
from app.services.trace_service import TraceService
from core.errors import DomainError, ErrorCode, ProviderError
from core.reasoning.base import InferenceBudget
from core.retrieval.base import RetrievalQuery
from core.tracing.base import TraceStage
from core.tracing.recorder import TraceCollector, TraceRecord
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


def _estimate_cost(settings: object, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in currency units, or None when no pricing is configured.

    A fabricated zero would silently understate what a benchmark run cost, so
    the absence of pricing is reported as absence.
    """
    input_rate = getattr(settings, "GENERATION_INPUT_COST_PER_MTOK", None)
    output_rate = getattr(settings, "GENERATION_OUTPUT_COST_PER_MTOK", None)
    if input_rate is None and output_rate is None:
        return None
    return round(
        input_tokens / 1_000_000 * (input_rate or 0.0)
        + output_tokens / 1_000_000 * (output_rate or 0.0),
        8,
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

    trace = TraceCollector()
    trace.set(strategy=str(payload.strategy), top_k=payload.top_k)
    embedding_model = (
        embeddings.model_id
        if embeddings is not None and payload.strategy in _NEEDS_EMBEDDINGS
        else None
    )

    # Bind the trace ID so every log line emitted during this query carries it
    # (spec section 48), making logs and the stored trace cross-referenceable.
    with bind_context(trace_id=str(trace.trace_id)):
        evidence: list[RetrievedChunk] = []
        try:
            evidence = await _retrieve(payload, project.id, session, embeddings, trace)

            async with trace.span(TraceStage.GENERATION, model=generator.model_id) as span:
                answer = await generator.generate(
                    payload.query,
                    evidence,
                    InferenceBudget(
                        max_input_tokens=0,
                        max_output_tokens=settings.GENERATION_MAX_OUTPUT_TOKENS,
                    ),
                )
                span.set(
                    claims=len(answer.claims),
                    abstained=answer.abstained,
                    invented_citations=answer.metadata.get("invented_citations", 0),
                    input_tokens=answer.usage.input_tokens,
                    output_tokens=answer.usage.output_tokens,
                )
            trace.record_usage(
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
                cost=_estimate_cost(
                    settings, answer.usage.input_tokens, answer.usage.output_tokens
                ),
            )
        except DomainError as exc:
            # Persist what did run before re-raising: a failed execution is
            # exactly the one worth being able to inspect afterwards.
            await _save_trace(
                session,
                trace.finish(error_code=str(exc.code), error_message=exc.message),
                project_id=project.id,
                principal=principal,
                payload=payload,
                model=generator.model_id,
                embedding_model=embedding_model,
                evidence_count=len(evidence),
                abstained=False,
                retrieved_chunk_ids=[hit.provenance.chunk_id for hit in evidence],
            )
            raise

        record = trace.finish()
        await _save_trace(
            session,
            record,
            project_id=project.id,
            principal=principal,
            payload=payload,
            model=answer.model_id,
            embedding_model=embedding_model,
            evidence_count=len(evidence),
            abstained=answer.abstained,
            answer=answer.text,
            # Stored so the evidence graph can be rebuilt from the trace alone
            # (spec section 18) rather than only from a live response.
            claims=[
                {"text": claim.text, "evidence": [str(c) for c in claim.evidence]}
                for claim in answer.claims
            ],
            retrieved_chunk_ids=[hit.provenance.chunk_id for hit in evidence],
            cited_chunk_ids=list(answer.cited_chunk_ids),
        )

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
        embedding_model=embedding_model,
        trace_id=record.trace_id,
        evidence=[to_evidence(hit) for hit in evidence] if payload.include_evidence else [],
        evidence_count=len(evidence),
        invented_citations=int(answer.metadata.get("invented_citations", 0)),
        unsupported_claims=len(answer.unsupported_claims),
        usage=QueryUsage(
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            cost=record.cost,
            retrieval_latency_ms=record.latency_of(TraceStage.RETRIEVAL) or 0.0,
            generation_latency_ms=record.latency_of(TraceStage.GENERATION) or 0.0,
            total_latency_ms=record.total_latency_ms,
        ),
    )


async def _retrieve(
    payload: QueryRequest,
    project_id: UUID,
    session: SessionDep,
    embeddings: EmbeddingsDep,
    trace: TraceCollector,
) -> list[RetrievedChunk]:
    """Run the chosen strategy inside a traced retrieval span."""
    query = RetrievalQuery(text=payload.query, project_id=project_id, top_k=payload.top_k)
    keyword = PostgresFtsRetriever(session)
    dense = PgVectorRetriever(session, embeddings)

    async with trace.span(TraceStage.RETRIEVAL, strategy=str(payload.strategy)) as span:
        match payload.strategy:
            case SearchStrategy.KEYWORD:
                evidence = await keyword.retrieve(query)
            case SearchStrategy.SEMANTIC:
                evidence = await dense.retrieve(query)
            case SearchStrategy.HYBRID:
                evidence = await HybridRetriever(dense, keyword).retrieve(query)
        span.set(
            candidate_count=len(evidence),
            # How many arms actually contributed, which is what makes an
            # empty keyword arm visible rather than invisible.
            sources=sorted(
                {str(item.retriever) for hit in evidence for item in (hit.contributions or ())}
            ),
        )
    return evidence


async def _save_trace(
    session: SessionDep,
    record: TraceRecord,
    *,
    project_id: UUID,
    principal: Principal,
    payload: QueryRequest,
    model: str | None,
    embedding_model: str | None,
    evidence_count: int,
    abstained: bool,
    answer: str | None = None,
    claims: list[dict[str, object]] | None = None,
    retrieved_chunk_ids: list[UUID] | None = None,
    cited_chunk_ids: list[UUID] | None = None,
) -> None:
    await TraceService(session).save(
        record,
        project_id=project_id,
        principal=principal,
        query_text=payload.query,
        strategy=str(payload.strategy),
        model=model,
        provider="mistral",
        embedding_model=embedding_model,
        evidence_count=evidence_count,
        abstained=abstained,
        answer=answer,
        claims=claims,
        retrieved_chunk_ids=retrieved_chunk_ids,
        cited_chunk_ids=cited_chunk_ids,
    )


@router.post("/stream")
async def stream_query(
    payload: QueryRequest, principal: PrincipalDep, session: SessionDep
) -> StreamingResponse:
    """Stream execution progress and answer tokens as Server-Sent Events.

    Unimplemented: streaming is spec section 45, and the stage events it would
    carry are exactly the trace spans this endpoint now records.
    """
    raise NotImplementedError
