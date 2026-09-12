"""Per-item scoring and aggregation (spec sections 26, 74).

Pure: takes what a run observed and returns numbers. Keeping it free of the
database and the pipeline means the metric definitions can be tested directly,
and reused by the benchmark and experiment runners without going through the
API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from evaluation.datasets.schema import EvaluationItemSpec
from evaluation.metrics import citation as citation_metrics
from evaluation.metrics import retrieval as retrieval_metrics
from evaluation.metrics.citation import ClaimCitations

# Reported at several depths because a system can look strong at k=10 and weak
# at k=1, and the difference is what reranking would address.
DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class ItemObservation:
    """Everything one item's execution produced."""

    retrieved: tuple[UUID, ...] = ()
    claims: tuple[ClaimCitations, ...] = ()
    answer: str | None = None
    abstained: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    error_code: str | None = None

    @property
    def cited(self) -> set[UUID]:
        return {chunk_id for claim in self.claims for chunk_id in claim.cited}


@dataclass(frozen=True, slots=True)
class ItemScore:
    """Metrics for one item. `None` means not measurable, never zero."""

    metrics: dict[str, float | None] = field(default_factory=dict)
    failure_category: str | None = None


def score_item(
    item: EvaluationItemSpec,
    observation: ItemObservation,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> ItemScore:
    """Score one item against its gold evidence."""
    gold = set(item.gold_chunks)
    retrieved = observation.retrieved
    metrics: dict[str, float | None] = {}

    for k in k_values:
        metrics[f"recall@{k}"] = retrieval_metrics.recall_at_k(retrieved, gold, k)
        metrics[f"precision@{k}"] = retrieval_metrics.precision_at_k(retrieved, gold, k)
        metrics[f"hit_rate@{k}"] = retrieval_metrics.hit_rate_at_k(retrieved, gold, k)
        metrics[f"ndcg@{k}"] = retrieval_metrics.ndcg_at_k(retrieved, gold, k)
    metrics["mrr"] = retrieval_metrics.reciprocal_rank(retrieved, gold)

    cited = observation.cited
    metrics["citation_precision"] = citation_metrics.citation_precision(cited, gold)
    metrics["citation_recall"] = citation_metrics.citation_recall(cited, gold)
    metrics["citation_validity"] = citation_metrics.citation_validity(observation.claims, retrieved)
    metrics["evidence_coverage"] = citation_metrics.evidence_coverage(cited, retrieved)
    metrics["unsupported_claim_rate"] = citation_metrics.unsupported_claim_rate(observation.claims)

    # Abstention is scored against what the item expects, so refusing on an
    # answerable question and answering an unanswerable one are both penalised.
    metrics["abstention_correct"] = float(observation.abstained == item.expects_abstention)

    metrics["input_tokens"] = float(observation.input_tokens)
    metrics["output_tokens"] = float(observation.output_tokens)
    metrics["total_tokens"] = float(observation.input_tokens + observation.output_tokens)
    metrics["cost"] = observation.cost
    metrics["latency_ms"] = observation.latency_ms
    metrics["retrieval_latency_ms"] = observation.retrieval_latency_ms
    metrics["generation_latency_ms"] = observation.generation_latency_ms
    metrics["failed"] = 1.0 if observation.error_code else 0.0

    return ItemScore(metrics=metrics, failure_category=_categorise(item, observation, metrics))


def _categorise(
    item: EvaluationItemSpec,
    observation: ItemObservation,
    metrics: dict[str, float | None],
) -> str | None:
    """Name the dominant failure, so failures can be counted by cause.

    Ordered most-fundamental first: retrieval failing explains a bad answer,
    whereas a bad answer over correct evidence is a different problem entirely
    (spec section 36).
    """
    if observation.error_code:
        return "execution_error"
    if item.gold_chunks and not (set(observation.retrieved) & set(item.gold_chunks)):
        return "retrieval_miss"
    if observation.abstained and not item.expects_abstention:
        return "over_abstention"
    if not observation.abstained and item.expects_abstention:
        return "answered_without_evidence"
    validity = metrics.get("citation_validity")
    if validity is not None and validity < 1.0:
        return "invented_citation"
    unsupported = metrics.get("unsupported_claim_rate")
    if unsupported is not None and unsupported > 0.0:
        return "unsupported_claim"
    return None


def aggregate(scores: Sequence[ItemScore]) -> dict[str, Any]:
    """Average each metric across items, ignoring where it was undefined.

    Every metric reports how many items contributed, because a recall@5 of
    0.9 over 2 of 50 items means something very different from 0.9 over 50 —
    and a bare average hides which one you are looking at.
    """
    names: list[str] = []
    for score in scores:
        for name in score.metrics:
            if name not in names:
                names.append(name)

    aggregated: dict[str, Any] = {}
    for name in names:
        values = [score.metrics.get(name) for score in scores]
        aggregated[name] = {
            "mean": retrieval_metrics.mean(values),
            "n": retrieval_metrics.defined_count(values),
        }

    categories: dict[str, int] = {}
    for score in scores:
        if score.failure_category:
            categories[score.failure_category] = categories.get(score.failure_category, 0) + 1
    aggregated["failure_categories"] = categories
    aggregated["item_count"] = len(scores)
    return aggregated


def aggregate_by(
    scores: Sequence[tuple[str, ItemScore]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per group — used to report per evidence condition.

    Pooling CLEAN with ADVERSARIAL hides exactly the contrast the platform
    exists to measure (spec section 28).
    """
    groups: dict[str, list[ItemScore]] = {}
    for key, score in scores:
        groups.setdefault(key, []).append(score)
    return {key: aggregate(items) for key, items in sorted(groups.items())}
