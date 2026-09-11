"""Project schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.common import APIModel, IdentifiedModel


class ProjectCreate(APIModel):
    """Payload for creating a project.

    `organization_id` is deliberately absent: ownership comes from the
    authenticated principal, never from client input.
    """

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    settings: dict[str, Any] = Field(default_factory=dict)


class ProjectRead(IdentifiedModel):
    """A project as returned by the API."""

    name: str
    description: str | None
    settings: dict[str, Any]
