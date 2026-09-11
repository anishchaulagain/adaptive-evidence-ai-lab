"""Reusable column types.

`Embedding` wraps pgvector so the dimensionality lives in one place and the
vector store can later be swapped for Qdrant (spec section 6).
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector


def Embedding(dimensions: int) -> Vector:  # noqa: N802 - factory reads as a type
    """A pgvector column of a fixed dimensionality."""
    return Vector(dimensions)
