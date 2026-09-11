"""Liveness and readiness probes against real dependencies."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_health_reports_version_and_environment(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["environment"]


async def test_readiness_checks_database_and_redis(client: AsyncClient) -> None:
    response = await client.get("/api/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"database": True, "redis": True}


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    """Spec section 48: a request ID must be traceable from the client side."""
    response = await client.get("/api/v1/health")

    assert response.headers["X-Request-ID"]


async def test_inbound_request_id_is_preserved(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "caller-supplied-id"})

    assert response.headers["X-Request-ID"] == "caller-supplied-id"


async def test_unknown_route_returns_the_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "NOT_FOUND"
    assert error["request_id"]
