"""Evaluation dataset vocabulary (spec sections 27, 28)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID


class EvidenceCondition(StrEnum):
    """The controlled evidence conditions (spec section 28).

    The point of the platform is that a system behaving well on CLEAN says
    little about how it behaves on CONFLICTING or ADVERSARIAL, so every item
    is labelled and results are reported per condition rather than pooled.
    """

    CLEAN = "clean"
    NOISY = "noisy"
    CONFLICTING = "conflicting"
    OUTDATED = "outdated"
    INCOMPLETE = "incomplete"
    ADVERSARIAL = "adversarial"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class EvaluationItemSpec:
    """One question and what a correct system would do with it.

    `gold_chunks` may legitimately be empty — an INCOMPLETE item is one whose
    answer is *not* in the corpus, where abstaining is the correct behaviour.
    Retrieval metrics report `None` for such items rather than zero.
    """

    question: str
    expected_answer: str | None = None
    gold_chunks: tuple[UUID, ...] = ()
    gold_documents: tuple[UUID, ...] = ()
    difficulty: Difficulty = Difficulty.MEDIUM
    evidence_condition: EvidenceCondition = EvidenceCondition.CLEAN
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def expects_abstention(self) -> bool:
        """Whether the correct answer is a refusal."""
        return self.evidence_condition is EvidenceCondition.INCOMPLETE
