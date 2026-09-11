"""Verification layer (spec section 21).

Checks the answer against the evidence that produced it: groundedness,
citation validity and contradiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.reasoning.base import GeneratedAnswer
from core.types import RetrievedChunk


@dataclass(frozen=True, slots=True)
class VerificationResult:
    grounded: bool
    groundedness_score: float
    unsupported_claims: tuple[str, ...] = ()
    invalid_citations: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class Verifier(Protocol):
    async def verify(
        self, answer: GeneratedAnswer, evidence: list[RetrievedChunk]
    ) -> VerificationResult: ...
