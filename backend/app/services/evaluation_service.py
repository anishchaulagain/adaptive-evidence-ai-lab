"""Evaluation run orchestration (spec section 26)."""

from __future__ import annotations

from app.services.base import Service


class EvaluationService(Service):
    """Run a dataset against a frozen configuration and record metrics.

    Results are always computed, never estimated (spec rule 9).
    """
