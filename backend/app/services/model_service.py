"""Model registry management (spec sections 7, 8)."""

from __future__ import annotations

from app.services.base import Service


class ModelService(Service):
    """Register, enable and describe models. Never returns credentials."""
