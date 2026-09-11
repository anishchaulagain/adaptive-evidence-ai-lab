"""Answer generation and adaptive inference budgeting (spec sections 20, 22)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from core.types import RetrievedChunk


@dataclass(frozen=True, slots=True)
class InferenceBudget:
    """The compute the router is willing to spend on this query."""

    max_input_tokens: int
    max_output_tokens: int
    allow_extended_thinking: bool = False
    max_retrieval_rounds: int = 1


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    citations: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


class AnswerGenerator(Protocol):
    async def generate(
        self, query: str, evidence: list[RetrievedChunk], budget: InferenceBudget
    ) -> GeneratedAnswer: ...

    def stream(
        self, query: str, evidence: list[RetrievedChunk], budget: InferenceBudget
    ) -> AsyncIterator[str]: ...
