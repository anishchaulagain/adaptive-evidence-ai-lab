"""Retriever selection by strategy.

One place that maps a strategy name onto a retriever, so `/search`, `/query`,
the evaluation runner and the comparison view cannot drift into building them
differently — which would make their numbers incomparable.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.adaptive import AdaptiveRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pgvector import PgVectorRetriever
from app.retrieval.postgres_fts import PostgresFtsRetriever
from core.embeddings.base import EmbeddingPipeline
from core.retrieval.base import Retriever

# Strategies that cannot run without an embedding provider.
EMBEDDING_STRATEGIES = frozenset({"semantic", "hybrid", "adaptive"})


def build_retriever(
    strategy: str,
    session: AsyncSession,
    embeddings: EmbeddingPipeline | None,
) -> Retriever:
    """Return the retriever for a strategy name."""
    keyword = PostgresFtsRetriever(session)
    dense = PgVectorRetriever(session, embeddings)

    match strategy:
        case "keyword":
            return keyword
        case "semantic":
            return dense
        case "adaptive":
            return AdaptiveRetriever(dense, keyword)
        case _:
            return HybridRetriever(dense, keyword)
