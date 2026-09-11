"""Adaptive query analyzer (spec section 14).

Prefer deterministic logic where it suffices - do not call an LLM for what
ordinary code can decide (spec rule 20).
"""

from __future__ import annotations

from typing import Protocol

from core.types import QueryAnalysis


class QueryAnalyzer(Protocol):
    def analyze(self, text: str) -> QueryAnalysis: ...
