"""Answer generation contracts (spec sections 20, 22).

Provider-independent: a chat model is reduced to "send messages, get JSON
back", so the prompt, the parsing and the citation rules live in `core` and
apply identically whichever provider answers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from core.types import RetrievedChunk


@dataclass(frozen=True, slots=True)
class InferenceBudget:
    """The compute the router is willing to spend on this query."""

    max_input_tokens: int
    max_output_tokens: int
    allow_extended_thinking: bool = False
    max_retrieval_rounds: int = 1


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion in an answer, with the evidence supporting it.

    Spec section 22 requires claim-level provenance: a single citation appended
    to a whole answer cannot be checked, because there is no way to tell which
    part of the answer it was meant to support.
    """

    text: str
    evidence: tuple[UUID, ...] = ()

    @property
    def is_supported(self) -> bool:
        """Whether any retrieved evidence was cited for this claim."""
        return bool(self.evidence)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """An answer and everything needed to audit it."""

    text: str
    claims: tuple[Claim, ...] = ()
    # True when the model reported that the evidence does not answer the query.
    # An honest refusal is a correct outcome, not a failure (spec rule 11).
    abstained: bool = False
    model_id: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cited_chunk_ids(self) -> tuple[UUID, ...]:
        """Every chunk cited anywhere in the answer, in first-cited order."""
        seen: dict[UUID, None] = {}
        for claim in self.claims:
            for chunk_id in claim.evidence:
                seen.setdefault(chunk_id, None)
        return tuple(seen)

    @property
    def unsupported_claims(self) -> tuple[Claim, ...]:
        """Claims that cite nothing — the answer's ungrounded assertions."""
        return tuple(claim for claim in self.claims if not claim.is_supported)


class ChatModel(Protocol):
    """A provider chat model that can return structured JSON."""

    model_id: str

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float,
    ) -> tuple[dict[str, Any], TokenUsage]:
        """Return the parsed JSON object and what the call consumed."""
        ...

    async def aclose(self) -> None: ...


class AnswerGenerator(Protocol):
    async def generate(
        self, query: str, evidence: list[RetrievedChunk], budget: InferenceBudget
    ) -> GeneratedAnswer: ...

    def stream(
        self, query: str, evidence: list[RetrievedChunk], budget: InferenceBudget
    ) -> AsyncIterator[str]: ...
