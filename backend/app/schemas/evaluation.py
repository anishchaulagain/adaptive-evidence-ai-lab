"""Evaluation schemas (spec sections 26, 27, 28)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel, IdentifiedModel
from app.schemas.search import SearchStrategy
from evaluation.datasets.schema import Difficulty, EvidenceCondition


class EvaluationItemCreate(APIModel):
    """One dataset item (spec section 27).

    `gold_chunks` may be empty. An INCOMPLETE item is one whose answer is not
    in the corpus, where abstaining is correct — retrieval metrics report
    `null` for it rather than a misleading zero.
    """

    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str | None = None
    gold_chunks: list[UUID] = Field(default_factory=list)
    gold_documents: list[UUID] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    evidence_condition: EvidenceCondition = EvidenceCondition.CLEAN


class EvaluationItemRead(IdentifiedModel):
    question: str
    expected_answer: str | None
    gold_chunks: list[UUID]
    gold_documents: list[UUID]
    difficulty: str
    evidence_condition: str


class DatasetCreate(APIModel):
    project_id: UUID
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    items: list[EvaluationItemCreate] = Field(min_length=1)


class DatasetRead(IdentifiedModel):
    project_id: UUID
    name: str
    description: str | None
    item_count: int
    items: list[EvaluationItemRead] = Field(default_factory=list)


class DatasetSummary(IdentifiedModel):
    project_id: UUID
    name: str
    description: str | None
    item_count: int


class RunCreate(APIModel):
    """A run request. The configuration is frozen onto the run at creation."""

    dataset_id: UUID
    name: str | None = None
    strategy: SearchStrategy = SearchStrategy.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    # Off by default: retrieval metrics need no provider, and generation costs
    # tokens on every item.
    generate: bool = False


class RunRead(IdentifiedModel):
    dataset_id: UUID
    project_id: UUID
    name: str | None
    config: dict[str, Any]
    status: str

    started_at: datetime | None
    finished_at: datetime | None

    item_count: int
    completed_count: int
    failed_count: int

    # `{"overall": {...}, "by_condition": {...}}`. Each metric reports its mean
    # and `n`, the number of items where it was defined — a mean over 2 of 50
    # items means something very different from a mean over 50.
    aggregate_metrics: dict[str, Any] = Field(default_factory=dict)

    error_code: str | None = None
    error_message: str | None = None


class ResultRead(APIModel):
    """What the system did on one item."""

    id: UUID
    item_id: UUID
    trace_id: UUID | None
    status: str
    answer: str | None
    abstained: bool
    retrieved_chunk_ids: list[UUID]
    cited_chunk_ids: list[UUID]
    metrics: dict[str, Any]
    failure_category: str | None
    error_code: str | None
