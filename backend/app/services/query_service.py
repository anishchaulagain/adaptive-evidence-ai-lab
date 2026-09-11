"""Query orchestration — the platform's central pipeline.

Sequences: query analysis -> adaptive hybrid retrieval -> fusion -> reranking
-> evidence assessment -> model routing -> generation -> verification, writing
a trace span at every stage (spec sections 14-23).
"""

from __future__ import annotations

from app.services.base import Service


class QueryService(Service):
    """Execute a query, buffered or streamed."""
