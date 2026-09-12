"""Embedding contracts.

`EmbeddingModel` is one provider-backed model — adapters implementing it live
in `models/providers/`. `EmbeddingPipeline` wraps a model with batching and
retry so callers never deal with provider limits.

Both protocols live here rather than in `models/` so that `core` defines the
contracts it depends on and never imports a provider.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingModel(Protocol):
    """A provider-backed embedding model."""

    model_id: str
    dimensions: int
    max_batch_size: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, returning one vector per input in the same order.

        Order is load-bearing: vectors are attached to chunks positionally, so
        a reordered response would silently mis-attribute every embedding.
        """
        ...

    async def aclose(self) -> None: ...


class EmbeddingPipeline(Protocol):
    dimensions: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
