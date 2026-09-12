"""Mistral adapter behaviour, against a mocked transport.

No network and no API key: httpx's MockTransport returns the responses, so
error mapping and retry policy are tested deterministically.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from core.errors import ErrorCode, ProviderError
from models.providers.mistral import (
    MAX_BATCH_SIZE,
    MISTRAL_EMBED_DIMENSIONS,
    MistralEmbeddingModel,
)

pytestmark = pytest.mark.unit


def _vector(seed: float = 0.1) -> list[float]:
    return [seed] * MISTRAL_EMBED_DIMENSIONS


def _ok_payload(count: int, *, reverse: bool = False) -> dict[str, Any]:
    indices = list(range(count))
    if reverse:
        indices.reverse()
    return {
        "data": [{"index": index, "embedding": _vector(0.1 * (index + 1))} for index in indices],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


def _model(handler: Any, **kwargs: Any) -> MistralEmbeddingModel:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.mistral.ai/v1",
    )
    return MistralEmbeddingModel(api_key="test-key", client=client, **kwargs)


# --- happy path -----------------------------------------------------------


async def test_embeds_a_batch() -> None:
    model = _model(lambda request: httpx.Response(200, json=_ok_payload(2)))

    vectors = await model.embed(["first", "second"])

    assert len(vectors) == 2
    assert all(len(vector) == MISTRAL_EMBED_DIMENSIONS for vector in vectors)


async def test_sends_the_configured_model_and_inputs() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_ok_payload(1))

    model = _model(handler)
    await model.embed(["only"])

    assert seen["model"] == "mistral-embed"
    assert seen["input"] == ["only"]
    assert seen["url"].endswith("/v1/embeddings")
    assert seen["auth"] == "Bearer test-key"


async def test_out_of_order_response_is_realigned() -> None:
    """The API returns an explicit index; vectors must be reordered by it or
    every chunk would silently get the wrong embedding."""
    model = _model(lambda request: httpx.Response(200, json=_ok_payload(3, reverse=True)))

    vectors = await model.embed(["a", "b", "c"])

    assert [round(vector[0], 3) for vector in vectors] == [0.1, 0.2, 0.3]


async def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made for an empty batch")

    assert await _model(handler).embed([]) == []


# --- error mapping --------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, ErrorCode.PROVIDER_NOT_CONFIGURED, False),
        (403, ErrorCode.PROVIDER_NOT_CONFIGURED, False),
        (422, ErrorCode.MODEL_CONTEXT_EXCEEDED, False),
        (429, ErrorCode.MODEL_RATE_LIMIT, True),
        (500, ErrorCode.MODEL_UNAVAILABLE, True),
        (503, ErrorCode.MODEL_UNAVAILABLE, True),
        (418, ErrorCode.EMBEDDING_FAILED, False),
    ],
)
async def test_http_status_maps_to_error_code(
    status: int, code: ErrorCode, retryable: bool
) -> None:
    model = _model(
        lambda request: httpx.Response(status, json={"message": "nope"}),
        max_retries=1,
    )

    with pytest.raises(ProviderError) as excinfo:
        await model.embed(["text"])

    assert excinfo.value.code is code
    assert excinfo.value.retryable is retryable
    assert excinfo.value.provider == "mistral"


async def test_error_details_do_not_leak_the_response_body() -> None:
    """`details` is serialised into API responses, and a provider body can echo
    the request — which is the user's document text."""
    model = _model(
        lambda request: httpx.Response(400, json={"message": "secret document text"}),
        max_retries=1,
    )

    with pytest.raises(ProviderError) as excinfo:
        await model.embed(["text"])

    assert "secret document text" not in str(excinfo.value.details)
    assert excinfo.value.details == {"status_code": 400}


async def test_timeout_maps_to_model_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ProviderError) as excinfo:
        await _model(handler, max_retries=1).embed(["text"])

    assert excinfo.value.code is ErrorCode.MODEL_TIMEOUT
    assert excinfo.value.retryable is True


# --- retry policy ---------------------------------------------------------


async def test_transient_failure_is_retried_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"message": "unavailable"})
        return httpx.Response(200, json=_ok_payload(1))

    vectors = await _model(handler, max_retries=3).embed(["text"])

    assert attempts["count"] == 2
    assert len(vectors) == 1


async def test_non_retryable_failure_is_not_retried() -> None:
    """Repeating a rejected key just burns the rate limit and hides the error."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(401, json={"message": "bad key"})

    with pytest.raises(ProviderError):
        await _model(handler, max_retries=3).embed(["text"])

    assert attempts["count"] == 1


# --- response validation --------------------------------------------------


async def test_wrong_vector_count_is_rejected() -> None:
    model = _model(lambda request: httpx.Response(200, json=_ok_payload(1)))

    with pytest.raises(ProviderError) as excinfo:
        await model.embed(["a", "b"])

    assert excinfo.value.code is ErrorCode.EMBEDDING_FAILED


async def test_wrong_dimensionality_is_rejected() -> None:
    """A vector of the wrong width would be rejected by the pgvector column
    much later; failing here names the real cause."""
    payload = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
    model = _model(lambda request: httpx.Response(200, json=payload))

    with pytest.raises(ProviderError) as excinfo:
        await model.embed(["text"])

    assert excinfo.value.code is ErrorCode.EMBEDDING_FAILED


async def test_malformed_response_is_rejected() -> None:
    model = _model(lambda request: httpx.Response(200, json={"unexpected": True}))

    with pytest.raises(ProviderError) as excinfo:
        await model.embed(["text"])

    assert excinfo.value.code is ErrorCode.EMBEDDING_FAILED


async def test_oversized_batch_is_refused_before_the_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("the batch should be refused before any request")

    with pytest.raises(ProviderError) as excinfo:
        await _model(handler).embed(["text"] * (MAX_BATCH_SIZE + 1))

    assert excinfo.value.code is ErrorCode.MODEL_CONTEXT_EXCEEDED
