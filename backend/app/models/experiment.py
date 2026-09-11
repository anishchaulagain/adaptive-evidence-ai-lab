"""Controlled experiments and their runs (spec sections 31-34)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class Experiment(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: name, hypothesis, config (frozen), conditions, seed, status."""

    __tablename__ = "experiments"


class ExperimentRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Columns: experiment_id, condition, evaluation_run_id, status, results."""

    __tablename__ = "experiment_runs"
