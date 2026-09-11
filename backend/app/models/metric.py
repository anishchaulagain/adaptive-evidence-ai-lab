"""Time-series system metrics (spec section 25)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class MetricRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: name, value, unit, recorded_at, dimensions (model, provider,
    stage, condition)."""

    __tablename__ = "metric_records"
