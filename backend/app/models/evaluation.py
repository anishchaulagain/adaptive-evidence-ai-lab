"""Evaluation lab: datasets, runs and per-item results (spec section 26)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class EvaluationDataset(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: name, description, item_count, schema_version."""

    __tablename__ = "evaluation_datasets"


class EvaluationItem(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Columns: dataset_id, question, reference_answer, relevant_chunk_ids,
    condition, difficulty."""

    __tablename__ = "evaluation_items"


class EvaluationRun(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: dataset_id, config (frozen, for reproducibility), status,
    started_at, finished_at, aggregate_metrics, error_code."""

    __tablename__ = "evaluation_runs"


class EvaluationResult(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Columns: run_id, item_id, trace_id, metrics, judge_verdicts,
    failure_category."""

    __tablename__ = "evaluation_results"
