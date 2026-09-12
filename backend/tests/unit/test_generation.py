"""Grounded answer generation and citation validation (spec section 22).

The rules that matter here are about what the model must *not* be able to do:
cite a passage that was never retrieved, or assert something with no citation
without that being visible.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from core.errors import ErrorCode, PipelineError
from core.reasoning.base import InferenceBudget
from core.reasoning.grounded import SYSTEM_PROMPT, GroundedAnswerGenerator, build_user_prompt
from core.types import Provenance, RetrievedChunk, RetrieverKind
from tests.fakes import FakeChatModel

pytestmark = pytest.mark.unit

DOCUMENT_ID = UUID("00000000-0000-0000-0000-0000000000aa")
BUDGET = InferenceBudget(max_input_tokens=0, max_output_tokens=512)


def _evidence(count: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=f"Passage number {index}.",
            score=1.0 - index / 100,
            retriever=RetrieverKind.HYBRID,
            provenance=Provenance(document_id=DOCUMENT_ID, chunk_id=uuid4()),
            rank=index,
        )
        for index in range(count)
    ]


async def _generate(
    payload: dict[str, Any], evidence: list[RetrievedChunk]
) -> tuple[Any, FakeChatModel]:
    model = FakeChatModel(payload)
    answer = await GroundedAnswerGenerator(model).generate("a question", evidence, BUDGET)
    return answer, model


# --- the happy path -------------------------------------------------------


async def test_claims_resolve_to_the_cited_chunks() -> None:
    evidence = _evidence(3)
    payload = {
        "answer": "Two things are true.",
        "claims": [
            {"text": "The first thing.", "evidence": [0]},
            {"text": "The second thing.", "evidence": [1, 2]},
        ],
    }

    answer, _ = await _generate(payload, evidence)

    assert answer.text == "Two things are true."
    assert [claim.text for claim in answer.claims] == ["The first thing.", "The second thing."]
    assert answer.claims[0].evidence == (evidence[0].provenance.chunk_id,)
    assert answer.claims[1].evidence == (
        evidence[1].provenance.chunk_id,
        evidence[2].provenance.chunk_id,
    )


async def test_usage_is_reported() -> None:
    answer, _ = await _generate({"answer": "x", "claims": []}, _evidence(1))

    assert answer.usage.input_tokens == 100
    assert answer.usage.output_tokens == 40
    assert answer.usage.total == 140


async def test_cited_chunk_ids_deduplicate_across_claims() -> None:
    evidence = _evidence(2)
    payload = {
        "answer": "a",
        "claims": [
            {"text": "one", "evidence": [0, 1]},
            {"text": "two", "evidence": [1]},
        ],
    }

    answer, _ = await _generate(payload, evidence)

    assert answer.cited_chunk_ids == (
        evidence[0].provenance.chunk_id,
        evidence[1].provenance.chunk_id,
    )


async def test_temperature_is_pinned_for_reproducibility() -> None:
    """A generator that varies between identical runs makes every downstream
    evaluation irreproducible."""
    _, model = await _generate({"answer": "x", "claims": []}, _evidence(1))

    assert model.calls[0]["temperature"] == 0.0


# --- citation validation --------------------------------------------------


async def test_an_invented_citation_is_dropped_and_counted() -> None:
    """The model must not be able to cite a passage that was never retrieved:
    a fabricated citation looks exactly as authoritative as a real one."""
    evidence = _evidence(2)
    payload = {
        "answer": "a",
        "claims": [{"text": "one", "evidence": [0, 7]}],
    }

    answer, _ = await _generate(payload, evidence)

    assert answer.claims[0].evidence == (evidence[0].provenance.chunk_id,)
    assert answer.metadata["invented_citations"] == 1


@pytest.mark.parametrize("cited", [[-1], ["two"], [None], [{"index": 0}], [1.5e9]], ids=str)
async def test_malformed_citations_are_rejected(cited: list[Any]) -> None:
    answer, _ = await _generate(
        {"answer": "a", "claims": [{"text": "one", "evidence": cited}]}, _evidence(2)
    )

    assert answer.claims[0].evidence == ()
    assert answer.metadata["invented_citations"] >= 1


async def test_duplicate_citations_collapse() -> None:
    evidence = _evidence(2)

    answer, _ = await _generate(
        {"answer": "a", "claims": [{"text": "one", "evidence": [0, 0, 0]}]}, evidence
    )

    assert answer.claims[0].evidence == (evidence[0].provenance.chunk_id,)


async def test_a_claim_citing_nothing_is_kept_and_flagged() -> None:
    """An unsupported claim is exactly what verification must be able to find,
    so it is surfaced rather than silently dropped."""
    payload = {
        "answer": "a",
        "claims": [
            {"text": "supported", "evidence": [0]},
            {"text": "ungrounded", "evidence": []},
        ],
    }

    answer, _ = await _generate(payload, _evidence(1))

    assert len(answer.claims) == 2
    assert [claim.is_supported for claim in answer.claims] == [True, False]
    assert [claim.text for claim in answer.unsupported_claims] == ["ungrounded"]


async def test_evidence_is_not_a_list_at_all() -> None:
    answer, _ = await _generate(
        {"answer": "a", "claims": [{"text": "one", "evidence": "zero"}]}, _evidence(1)
    )

    assert answer.claims[0].evidence == ()


# --- abstention -----------------------------------------------------------


async def test_no_evidence_abstains_without_calling_the_model() -> None:
    """There is nothing to answer from, and paying for a request to be told so
    is waste."""
    model = FakeChatModel({"answer": "should not be used", "claims": []})

    answer = await GroundedAnswerGenerator(model).generate("a question", [], BUDGET)

    assert answer.abstained is True
    assert answer.claims == ()
    assert model.calls == [], "the model must not be called with no evidence"
    assert answer.metadata["reason"] == "no_evidence"


async def test_model_abstention_is_preserved() -> None:
    answer, _ = await _generate(
        {"answer": "The passages do not say.", "claims": [], "abstained": True},
        _evidence(2),
    )

    assert answer.abstained is True
    assert answer.text == "The passages do not say."


async def test_an_empty_answer_that_does_not_abstain_is_an_error() -> None:
    """Silence without an explicit refusal is a generation failure, not a
    valid answer."""
    with pytest.raises(PipelineError) as excinfo:
        await _generate({"answer": "   ", "claims": []}, _evidence(1))

    assert excinfo.value.code is ErrorCode.GENERATION_FAILED
    assert excinfo.value.stage == "generation"


# --- malformed model output -----------------------------------------------


async def test_a_response_without_an_answer_key_is_an_error() -> None:
    with pytest.raises(PipelineError) as excinfo:
        await _generate({"claims": []}, _evidence(1))

    assert excinfo.value.code is ErrorCode.GENERATION_FAILED


@pytest.mark.parametrize(
    "claims", [None, "not a list", [None], ["text"], [{"text": ""}], [{}]], ids=str
)
async def test_malformed_claims_are_skipped_not_fatal(claims: Any) -> None:
    """Losing a malformed claim is recoverable; discarding a whole answer over
    one is not."""
    answer, _ = await _generate({"answer": "still useful", "claims": claims}, _evidence(1))

    assert answer.text == "still useful"


# --- prompt construction --------------------------------------------------


def test_the_prompt_numbers_passages_positionally() -> None:
    """Indices, not UUIDs: they cost fewer tokens and, unlike an invented UUID,
    an out-of-range index is detectable."""
    prompt = build_user_prompt("why?", _evidence(3))

    assert "Question: why?" in prompt
    assert "[0] Passage number 0." in prompt
    assert "[2] Passage number 2." in prompt


def test_long_passages_are_truncated() -> None:
    huge = RetrievedChunk(
        text="x" * 10_000,
        score=1.0,
        retriever=RetrieverKind.SEMANTIC,
        provenance=Provenance(document_id=DOCUMENT_ID, chunk_id=uuid4()),
        rank=0,
    )

    prompt = build_user_prompt("why?", [huge])

    assert len(prompt) < 10_000


def test_the_system_prompt_forbids_outside_knowledge() -> None:
    assert "only the passages" in SYSTEM_PROMPT.lower()
    assert "abstained" in SYSTEM_PROMPT
