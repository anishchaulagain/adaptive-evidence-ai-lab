"""Query request/response schemas (spec sections 13, 17, 22).

The response is deliberately verbose. This platform exists to make an answer
auditable, so what was retrieved, what was cited, which model ran and what it
consumed are all part of the contract rather than internal detail.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel
from app.schemas.search import EvidenceItem, SearchStrategy


class QueryRequest(APIModel):
    """A question asked against one project's evidence."""

    project_id: UUID
    query: str = Field(min_length=1, max_length=4000)
    strategy: SearchStrategy = SearchStrategy.HYBRID
    # How much evidence reaches the model. Retrieval may fetch more than this.
    top_k: int = Field(default=8, ge=1, le=50)
    # Return the retrieved evidence alongside the answer. On by default: an
    # answer whose evidence cannot be inspected is not auditable.
    include_evidence: bool = True


class CitationRef(APIModel):
    """A resolved citation, pointing at an exact span of a document."""

    chunk_id: UUID
    document_id: UUID
    page: int | None
    char_start: int | None
    char_end: int | None
    text: str


class ClaimRead(APIModel):
    """One assertion in the answer with the evidence supporting it.

    `evidence` empty means the model asserted something it cited nothing for —
    kept rather than hidden, because an unsupported claim is exactly what the
    verification layer must be able to find.
    """

    text: str
    evidence: list[CitationRef] = Field(default_factory=list)
    supported: bool


class QueryUsage(APIModel):
    """What producing this answer consumed."""

    input_tokens: int
    output_tokens: int
    # Null unless token pricing is configured; never a fabricated zero.
    cost: float | None = None
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class QueryResponse(APIModel):
    """An answer, its provenance, and what it cost to produce."""

    query: str
    answer: str
    # True when the model reported the evidence does not answer the question.
    # An honest refusal is a correct outcome, not a failure.
    abstained: bool
    claims: list[ClaimRead] = Field(default_factory=list)

    strategy: SearchStrategy
    model: str
    embedding_model: str | None = None
    # The persisted execution trace for this answer. Fetch it from
    # /traces/{id} to see every stage, its timing and its metadata.
    trace_id: UUID

    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_count: int
    # Citations the model produced that pointed at no retrieved passage. A
    # non-zero count is a groundedness signal, not a parsing detail.
    invented_citations: int = 0
    unsupported_claims: int = 0

    usage: QueryUsage
