"""Trace persistence and retrieval (spec section 23)."""

from __future__ import annotations

from app.services.base import Service


class TraceService(Service):
    """Persist trace spans and assemble traces for the trace viewer."""
