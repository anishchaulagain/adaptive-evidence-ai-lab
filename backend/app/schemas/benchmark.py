"""Benchmark schemas (spec sections 29, 30)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class BenchmarkRunCreate(APIModel):
    """Fields: suite, conditions, config, seed."""


class BenchmarkRunRead(IdentifiedModel):
    """Fields: suite, condition, config, seed, code_version, status, results."""
