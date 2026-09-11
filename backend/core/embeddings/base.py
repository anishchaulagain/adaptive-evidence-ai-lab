"""Embedding pipeline interface.

Wraps an embedding model adapter (`models.embeddings`) with batching, caching
and retry so callers do not deal with provider rate limits.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingPipeline(Protocol):
    dimensions: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
