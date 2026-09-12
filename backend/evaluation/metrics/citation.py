"""Citation and reliability metrics (spec section 26).

Deterministic: these compare what the answer cited against the gold evidence
and against what was actually retrieved. No LLM judge is involved, so they are
reproducible and free — unlike faithfulness and answer correctness, which need
a judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ClaimCitations:
    """One claim's citations, as produced by the generator."""

    text: str
    cited: tuple[UUID, ...]


def citation_precision(cited: set[UUID], gold: set[UUID]) -> float | None:
    """Fraction of cited chunks that are genuinely relevant.

    Low precision means the answer padded its citations with passages that do
    not support it — which reads as well-sourced while not being.
    """
    if not gold:
        return None
    if not cited:
        return None
    return len(cited & gold) / len(cited)


def citation_recall(cited: set[UUID], gold: set[UUID]) -> float | None:
    """Fraction of gold chunks the answer actually cited."""
    if not gold:
        return None
    return len(cited & gold) / len(gold)


def evidence_coverage(cited: set[UUID], retrieved: Sequence[UUID]) -> float | None:
    """Fraction of the retrieved evidence the answer used.

    Not a quality score in itself. Very low coverage means most of what was
    retrieved, paid for and sent to the model went unused, which is a cost
    signal for the Phase 14 budget experiments.
    """
    if not retrieved:
        return None
    return len(cited & set(retrieved)) / len(set(retrieved))


def unsupported_claim_rate(claims: Sequence[ClaimCitations]) -> float | None:
    """Fraction of claims that cite nothing at all.

    An uncited claim is an assertion the answer makes on its own authority,
    which is precisely what a grounded system must not do.
    """
    if not claims:
        return None
    return sum(1 for claim in claims if not claim.cited) / len(claims)


def citation_validity(claims: Sequence[ClaimCitations], retrieved: Sequence[UUID]) -> float | None:
    """Fraction of citations that point at something actually retrieved.

    Below 1.0 means the generator invented a citation. The API drops those
    before they reach a caller, so this measures the model's behaviour rather
    than what survived validation.
    """
    total = sum(len(claim.cited) for claim in claims)
    if total == 0:
        return None
    available = set(retrieved)
    valid = sum(1 for claim in claims for chunk_id in claim.cited if chunk_id in available)
    return valid / total
