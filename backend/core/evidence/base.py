"""Evidence assessment and the evidence graph (spec section 18)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.types import RetrievedChunk


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """How trustworthy the retrieved set is, before generation is attempted."""

    sufficiency: float
    agreement: float
    coverage: float
    conflict_detected: bool
    uncertainty: float


class EvidenceAssessor(Protocol):
    def assess(self, query: str, chunks: list[RetrievedChunk]) -> EvidenceAssessment: ...
