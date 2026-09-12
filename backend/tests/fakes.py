"""Test doubles.

A deterministic embedder lets the retrieval path be exercised end to end with
no provider key and no network, while still producing vectors where lexically
similar texts are genuinely close — so ranking assertions mean something.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.models.chunk import EMBEDDING_DIMENSIONS

_TOKEN = re.compile(r"[a-z0-9]+")


def deterministic_vector(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Hash text into a unit vector.

    Bag-of-words hashing: shared vocabulary moves two texts closer, so cosine
    similarity behaves the way the real model's does for the purposes of
    ranking tests. Identical text always yields an identical vector.
    """
    vector = [0.0] * dimensions
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        # Sign from a second byte, so unrelated tokens can cancel rather than
        # only ever accumulating in the same direction.
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # An empty or symbol-only string still needs a valid unit vector.
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeEmbeddingModel:
    """Implements `EmbeddingModel` without any network access."""

    def __init__(
        self,
        *,
        model_id: str = "fake-embed",
        dimensions: int = EMBEDDING_DIMENSIONS,
        max_batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self.calls: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > self.max_batch_size:
            raise AssertionError(
                f"pipeline sent {len(texts)} texts, exceeding max_batch_size {self.max_batch_size}"
            )
        self.calls.append(list(texts))
        return [deterministic_vector(text, self.dimensions) for text in texts]

    async def aclose(self) -> None:
        self.closed = True
