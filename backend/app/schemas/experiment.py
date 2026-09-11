"""Experiment schemas (spec sections 33, 34)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class ExperimentCreate(APIModel):
    """Fields: name, hypothesis, conditions, config, seed, dataset_id."""


class ExperimentRead(IdentifiedModel):
    """Fields: name, hypothesis, conditions, config, seed, status, results."""
