"""Query request/response schemas (spec sections 13, 17, 22)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class RetrievalConfig(APIModel):
    """Fields: strategy (semantic|keyword|hybrid|adaptive), top_k, alpha,
    rerank, rerank_top_k, filters."""


class QueryRequest(APIModel):
    """Fields: project_id, text, retrieval (RetrievalConfig), model_config_id,
    inference_budget, stream."""


class Citation(APIModel):
    """Fields: chunk_id, document_id, page, char_start, char_end, quote, score."""


class EvidenceItem(APIModel):
    """One retrieved chunk with its provenance (spec section 17).

    Fields: chunk_id, text, score, retriever (semantic|keyword), rank_before,
    rank_after, rerank_score, document_title, page.
    """


class QueryResponse(IdentifiedModel):
    """Fields: answer, citations, evidence, model_used, routing_reason,
    trace_id, tokens, cost, latency_ms, verification."""
