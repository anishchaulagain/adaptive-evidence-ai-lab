"""Registered models and their runtime configuration (spec section 8)."""

from __future__ import annotations

from app.db.base import Base, TenantMixin, TimestampMixin, UUIDPrimaryKeyMixin


class ModelConfig(Base, UUIDPrimaryKeyMixin, TimestampMixin, TenantMixin):
    """Columns: name, provider, model_id, role, parameters, cost_per_input_token,
    cost_per_output_token, context_window, capabilities, is_enabled."""

    __tablename__ = "model_configs"
