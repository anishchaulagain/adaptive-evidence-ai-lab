"""Experiment orchestration (spec sections 31-34)."""

from __future__ import annotations

from app.services.base import Service


class ExperimentService(Service):
    """Run an experiment across conditions with a fixed seed and frozen config."""
