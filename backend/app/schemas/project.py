"""Project schemas."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class ProjectCreate(APIModel):
    """Fields: name, description, settings."""


class ProjectRead(IdentifiedModel):
    """Fields: name, description, settings, counts (sources, documents, chunks)."""
