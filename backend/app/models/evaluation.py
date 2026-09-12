"""Evaluation lab: datasets, runs and per-item results (spec section 26)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    ARRAY,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class EvaluationDataset(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """A set of questions with known-good answers and evidence."""

    __tablename__ = "evaluation_datasets"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_evaluation_datasets_project_id_name"),
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    items: Mapped[list[EvaluationItem]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="EvaluationItem.created_at",
        lazy="selectin",
    )


class EvaluationItem(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One question, its expected answer and its gold evidence.

    Gold IDs are typed UUID arrays rather than foreign keys: a dataset should
    survive its corpus being re-ingested or a document being deleted, and a
    dangling gold reference is a measurable fact about a run, not a reason to
    refuse to store the dataset. Chunk IDs are derived deterministically from
    document and ordinal, so a gold set survives re-ingestion.
    """

    __tablename__ = "evaluation_items"

    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    gold_chunk_ids: Mapped[list[UUID]] = mapped_column(ARRAY(Uuid), nullable=False, default=list)
    gold_document_ids: Mapped[list[UUID]] = mapped_column(ARRAY(Uuid), nullable=False, default=list)

    difficulty: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")
    evidence_condition: Mapped[str] = mapped_column(
        String(24), nullable=False, default="clean", index=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", nullable=False, default=dict)

    dataset: Mapped[EvaluationDataset] = relationship(back_populates="items")


class EvaluationRun(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """One execution of a dataset against a frozen configuration.

    `config` is a snapshot, not a reference: a run's meaning must not change
    because a project's defaults were edited afterwards (spec principle 2).
    """

    __tablename__ = "evaluation_runs"

    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    aggregate_metrics: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )


class EvaluationResult(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """What the system did on one item of one run."""

    __tablename__ = "evaluation_results"

    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    item_id: Mapped[UUID] = mapped_column(
        ForeignKey("evaluation_items.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Links a measurement to the execution that produced it, so a surprising
    # score can be opened as a full trace rather than re-run to investigate.
    trace_id: Mapped[UUID | None] = mapped_column(nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstained: Mapped[bool] = mapped_column(nullable=False, default=False)

    retrieved_chunk_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(Uuid), nullable=False, default=list
    )
    cited_chunk_ids: Mapped[list[UUID]] = mapped_column(ARRAY(Uuid), nullable=False, default=list)

    metrics: Mapped[dict[str, Any]] = mapped_column(nullable=False, default=dict)
    failure_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")
