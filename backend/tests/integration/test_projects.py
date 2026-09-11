"""Projects vertical slice: route -> dependency -> service -> ORM -> Postgres."""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("clean_projects")]

ENDPOINT = "/api/v1/projects"


async def test_create_project_persists_and_returns_it(client: AsyncClient) -> None:
    response = await client.post(
        ENDPOINT,
        json={"name": "Climate Reports", "description": "IPCC corpus"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Climate Reports"
    assert body["description"] == "IPCC corpus"
    assert body["settings"] == {}
    assert body["id"]
    assert body["created_at"]

    listed = await client.get(ENDPOINT)
    assert [project["id"] for project in listed.json()] == [body["id"]]


async def test_created_project_is_readable_by_id(client: AsyncClient) -> None:
    created = await client.post(ENDPOINT, json={"name": "Readable"})
    project_id = created.json()["id"]

    response = await client.get(f"{ENDPOINT}/{project_id}")

    assert response.status_code == 200
    assert response.json()["id"] == project_id


async def test_settings_round_trip_through_jsonb(client: AsyncClient) -> None:
    """`settings` is JSONB; nested structures must survive the round trip."""
    payload = {"retrieval": {"strategy": "hybrid", "top_k": 20}, "tags": ["a", "b"]}

    created = await client.post(ENDPOINT, json={"name": "Configured", "settings": payload})
    fetched = await client.get(f"{ENDPOINT}/{created.json()['id']}")

    assert fetched.json()["settings"] == payload


async def test_duplicate_name_in_one_organization_conflicts(client: AsyncClient) -> None:
    await client.post(ENDPOINT, json={"name": "Duplicate"})

    response = await client.post(ENDPOINT, json={"name": "Duplicate"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONFLICT"


async def test_unknown_project_is_not_found(client: AsyncClient) -> None:
    response = await client.get(f"{ENDPOINT}/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


async def test_malformed_id_is_a_validation_error(client: AsyncClient) -> None:
    response = await client.get(f"{ENDPOINT}/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": ""},
        {"name": "x" * 201},
        {"name": "Valid", "unexpected_field": "rejected"},
    ],
    ids=["missing-name", "empty-name", "name-too-long", "unknown-field"],
)
async def test_invalid_payloads_are_rejected(
    client: AsyncClient, payload: dict[str, object]
) -> None:
    response = await client.post(ENDPOINT, json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_ownership_is_taken_from_the_principal_not_the_payload(
    client: AsyncClient,
) -> None:
    """A client must not be able to create a project in another organization."""
    response = await client.post(
        ENDPOINT,
        json={"name": "Injected", "organization_id": str(uuid4())},
    )

    assert response.status_code == 422
