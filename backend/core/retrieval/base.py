"""Retrieval interfaces.

The interface stays abstract so pgvector can be swapped for Qdrant, and
Postgres FTS for a dedicated BM25 implementation (spec section 6), without
touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from core.types import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    project_id: UUID
    top_k: int = 10
    filters: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Every retriever reports which strategy produced each hit, so retrieval
    methods can be compared (spec section 15)."""

    name: str

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


class VectorStore(Protocol):
    """Vector index abstraction. Implementations: pgvector, Qdrant."""

    async def upsert(
        self, project_id: UUID, ids: list[UUID], vectors: list[list[float]]
    ) -> None: ...

    async def search(
        self, project_id: UUID, vector: list[float], top_k: int
    ) -> list[tuple[UUID, float]]: ...

    async def delete(self, project_id: UUID, ids: list[UUID]) -> None: ...
