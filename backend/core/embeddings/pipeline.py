"""Batched embedding pipeline.

Sits between callers and a provider adapter: splits work into batches the
provider accepts, and preserves input order so vectors stay attached to the
right chunks.
"""

from __future__ import annotations

from core.embeddings.base import EmbeddingModel


class BatchedEmbeddingPipeline:
    """Implements `EmbeddingPipeline` over any `EmbeddingModel`."""

    def __init__(self, model: EmbeddingModel, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model = model
        # Never exceed what the provider accepts, whatever configuration says.
        self._batch_size = min(batch_size, model.max_batch_size)
        self.dimensions = model.dimensions

    @property
    def model_id(self) -> str:
        return self._model.model_id

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning vectors in the input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._model.embed(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Uses the same model as the corpus: a query embedded by a different
        model would not share a vector space with the chunks, and similarity
        scores would be meaningless.
        """
        vectors = await self._model.embed([text])
        return vectors[0]

    async def aclose(self) -> None:
        await self._model.aclose()
