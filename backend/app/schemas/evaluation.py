"""Evaluation schemas (spec section 26)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class EvaluationRunCreate(APIModel):
    """Fields: dataset_id, config (retrieval + model + budget), conditions."""


class EvaluationMetrics(APIModel):
    """Retrieval, generation, citation, reliability and systems metrics.

    Fields: recall_at_k, precision_at_k, mrr, ndcg, faithfulness,
    answer_relevance, citation_precision, citation_recall, abstention_rate,
    latency_ms, tokens, cost.
    """


class EvaluationRunRead(IdentifiedModel):
    """Fields: dataset_id, config, status, started_at, finished_at,
    aggregate_metrics, error_code."""
