"""Shared response primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Base for every schema: ORM-friendly and strict about unknown fields."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class IdentifiedModel(APIModel):
    id: UUID
    created_at: datetime
    updated_at: datetime


class HealthResponse(APIModel):
    status: Literal["ok"]
    version: str
    environment: str


class ReadinessResponse(APIModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorDetail(APIModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    trace_id: str | None = None


class ErrorResponse(APIModel):
    error: ErrorDetail


class Page[T](APIModel):
    """Cursor-free offset pagination; sufficient for a research console."""

    items: list[T]
    total: int
    limit: int
    offset: int
