"""Shared test fixtures.

Integration tests run against a real Postgres and Redis — the same engines
production uses — because the things most likely to break here (JSONB columns,
foreign keys, transaction scoping) are exactly what an in-memory substitute
would fake.

The schema is built by running the real migrations, so a broken migration
fails the suite rather than hiding behind `create_all()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
from alembic.config import Config
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from alembic import command
from app.core.config import Settings, get_settings
from app.core.security import DEV_PRINCIPAL
from app.db.session import get_sessionmaker

TEST_DATABASE_SUFFIX = "_test"


def _test_database_url(url: str) -> tuple[str, str, str]:
    """Return (test_url, admin_url, test_database_name)."""
    parsed = urlparse(url)
    name = parsed.path.lstrip("/")
    test_name = f"{name}{TEST_DATABASE_SUFFIX}"
    test_url = urlunparse(parsed._replace(path=f"/{test_name}"))
    admin_url = urlunparse(parsed._replace(path="/postgres"))
    return test_url, admin_url, test_name


async def _create_database(admin_url: str, name: str) -> None:
    """Create the test database if it does not exist."""
    dsn = admin_url.replace("postgresql+asyncpg://", "postgresql://")
    connection = await asyncpg.connect(dsn)
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await connection.execute(f'CREATE DATABASE "{name}"')
    finally:
        await connection.close()


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Application settings pointed at a dedicated test database."""
    base = get_settings()
    test_url, _, _ = _test_database_url(str(base.DATABASE_URL))
    return base.model_copy(update={"DATABASE_URL": test_url})


@pytest.fixture(scope="session", autouse=True)
def _migrated_database(settings: Settings) -> None:
    """Create the test database and bring it to head before any test runs."""
    test_url, admin_url, name = _test_database_url(str(get_settings().DATABASE_URL))
    asyncio.run(_create_database(admin_url, name))

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", test_url)
    command.upgrade(config, "head")


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    """A fresh application instance built from test settings."""
    from app.main import create_app

    application = create_app(settings)
    # Dependencies resolve settings through `get_settings`; point them at the
    # test instance rather than the process environment.
    application.dependency_overrides[get_settings] = lambda: settings
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An ASGI-transport client. Runs the real lifespan: no network, no server."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest.fixture
async def clean_projects(client: AsyncClient) -> AsyncIterator[None]:
    """Isolate a test from project rows left by any other test.

    Truncating on both sides matters: assertions about the full project
    listing would otherwise depend on test execution order. The cascade also
    removes dependent documents and chunks.
    """
    await _truncate_projects()
    yield
    await _truncate_projects()


async def _truncate_projects() -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await session.execute(text("TRUNCATE TABLE projects CASCADE"))
        await session.commit()


@pytest.fixture
def principal() -> object:
    """The development principal the API runs as while AUTH_MODE=disabled."""
    return DEV_PRINCIPAL
