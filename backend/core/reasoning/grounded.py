"""Grounded answer generation with claim-level citations (spec section 22).

Evidence is presented to the model by positional index, never by chunk ID. Two
reasons: a UUID wastes tokens and invites transcription errors, and an index
can be validated against what was actually retrieved — so a citation the model
invents is detectable rather than plausible-looking.

Provider-independent: it needs only a `ChatModel`.
"""

from __future__ import annotations

from typing import Any

from core.errors import ErrorCode, PipelineError
from core.reasoning.base import (
    ChatModel,
    Claim,
    GeneratedAnswer,
    InferenceBudget,
    TokenUsage,
)
from core.types import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions strictly from numbered passages supplied by the user.

Rules:
- Use only the passages. Never use outside knowledge, and never guess.
- Break your answer into claims. Every claim must cite the passage indices \
that support it.
- A claim with no supporting passage must not be made.
- If the passages do not answer the question, set "abstained" to true and \
leave "claims" empty. Refusing is correct when the evidence is absent.

Respond with JSON only, in this exact shape:
{"answer": "<the full answer as prose>",
 "claims": [{"text": "<one assertion>", "evidence": [<passage index>, ...]}],
 "abstained": false}\
"""

# Guard against one enormous chunk crowding out every other passage.
MAX_PASSAGE_CHARS = 4000


def build_user_prompt(query: str, evidence: list[RetrievedChunk]) -> str:
    """Render the query and numbered passages."""
    lines = [f"Question: {query}", "", "Passages:"]
    for index, chunk in enumerate(evidence):
        lines.append(f"[{index}] {chunk.text[:MAX_PASSAGE_CHARS]}")
    if not evidence:
        lines.append("(none)")
    return "\n".join(lines)


class GroundedAnswerGenerator:
    """Implements `AnswerGenerator` over any `ChatModel`."""

    name = "grounded_json"

    def __init__(self, model: ChatModel, *, temperature: float = 0.0) -> None:
        self._model = model
        # Pinned to 0 by default: an answer that changes between identical runs
        # would make every downstream evaluation irreproducible (spec rule 19).
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model.model_id

    async def aclose(self) -> None:
        """Release the model's connection pool."""
        await self._model.aclose()

    async def generate(
        self, query: str, evidence: list[RetrievedChunk], budget: InferenceBudget
    ) -> GeneratedAnswer:
        """Answer the query from the evidence, citing passages per claim.

        With no evidence the model is never called: there is nothing it could
        honestly answer from, and spending a request to be told so is waste.
        """
        if not evidence:
            return GeneratedAnswer(
                text="No relevant evidence was retrieved for this question.",
                abstained=True,
                model_id=self._model.model_id,
                metadata={"reason": "no_evidence"},
            )

        payload, usage = await self._model.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(query, evidence),
            max_output_tokens=budget.max_output_tokens,
            temperature=self._temperature,
        )
        return self._parse(payload, evidence, usage)

    def _parse(
        self,
        payload: dict[str, Any],
        evidence: list[RetrievedChunk],
        usage: TokenUsage,
    ) -> GeneratedAnswer:
        if not isinstance(payload, dict) or "answer" not in payload:
            raise PipelineError(
                "The model did not return an answer object.",
                code=ErrorCode.GENERATION_FAILED,
                stage="generation",
            )

        text = str(payload.get("answer") or "").strip()
        abstained = bool(payload.get("abstained", False))

        claims: list[Claim] = []
        dropped = 0
        for raw in payload.get("claims") or []:
            if not isinstance(raw, dict):
                continue
            claim_text = str(raw.get("text") or "").strip()
            if not claim_text:
                continue
            cited, invented = self._resolve_evidence(raw.get("evidence"), evidence)
            dropped += invented
            claims.append(Claim(text=claim_text, evidence=cited))

        if not text and not abstained:
            raise PipelineError(
                "The model returned an empty answer.",
                code=ErrorCode.GENERATION_FAILED,
                stage="generation",
            )

        return GeneratedAnswer(
            text=text,
            claims=tuple(claims),
            abstained=abstained,
            model_id=self._model.model_id,
            usage=usage,
            metadata={
                "evidence_count": len(evidence),
                # Surfaced rather than silently swallowed: a model citing
                # passages that were never retrieved is a groundedness failure
                # the verification layer needs to see.
                "invented_citations": dropped,
            },
        )

    def _resolve_evidence(
        self, raw_indices: Any, evidence: list[RetrievedChunk]
    ) -> tuple[tuple[Any, ...], int]:
        """Map cited indices to chunk IDs, dropping any that were not retrieved.

        Returns the resolved chunk IDs and how many citations were invented.
        """
        if not isinstance(raw_indices, list):
            return (), 0

        resolved: list[Any] = []
        invented = 0
        seen: set[int] = set()
        for value in raw_indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                invented += 1
                continue
            if not 0 <= index < len(evidence):
                invented += 1
                continue
            if index in seen:
                continue
            seen.add(index)
            resolved.append(evidence[index].provenance.chunk_id)
        return tuple(resolved), invented
