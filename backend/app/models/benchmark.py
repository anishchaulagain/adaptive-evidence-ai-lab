"""AE-Bench runs (spec section 29)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class BenchmarkRun(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: suite, condition, config (frozen), seed, code_version, status,
    started_at, finished_at, results."""

    __tablename__ = "benchmark_runs"
