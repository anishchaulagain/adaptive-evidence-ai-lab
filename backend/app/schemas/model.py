"""Model configuration schemas (spec section 8)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class ModelConfigCreate(APIModel):
    """Fields: name, provider, model_id, role, parameters.

    API keys are supplied through the environment, never through this payload.
    """


class ModelConfigRead(IdentifiedModel):
    """Fields: name, provider, model_id, role, parameters, context_window,
    cost_per_input_token, cost_per_output_token, capabilities, is_enabled."""
