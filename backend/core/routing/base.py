"""Model routing (spec section 19).

Routing decisions must be explainable: every decision records why a model was
chosen, which the UI surfaces in the model routing graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.evidence.base import EvidenceAssessment
from core.types import QueryAnalysis


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    model_id: str
    provider: str
    reason: str
    estimated_cost: float | None = None
    estimated_latency_ms: float | None = None


class ModelRouter(Protocol):
    def route(
        self, analysis: QueryAnalysis, evidence: EvidenceAssessment
    ) -> RoutingDecision: ...
