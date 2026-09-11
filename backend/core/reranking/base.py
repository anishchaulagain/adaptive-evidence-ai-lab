"""Reranking interface (spec section 15)."""

from __future__ import annotations

from typing import Protocol

from core.types import RetrievedChunk


class Reranker(Protocol):
    """Reranking must be measurable: implementations record the rank change so
    its usefulness can be evaluated rather than assumed."""

    name: str

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]: ...
