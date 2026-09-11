"""Shared test fixtures.

Integration fixtures are intentionally unimplemented: the database and Redis
wiring is decided in Phase 1.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings for tests. Override individual values per test as needed."""
    raise NotImplementedError


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    """A fresh application instance built from test settings."""
    raise NotImplementedError


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An ASGI-transport client; no network, no running server."""
    raise NotImplementedError
