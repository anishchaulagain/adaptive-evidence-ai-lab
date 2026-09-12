"""Mistral provider adapter.

Talks to the REST API over httpx rather than the vendor SDK: the surface used
here is a handful of endpoints, and a direct client keeps error mapping,
timeouts and retry policy explicit and testable.

All provider-specific knowledge is confined to this module (spec rule 6).
`MistralTransport` holds the HTTP concerns so every capability — embeddings
and generation — shares one error taxonomy and one retry policy rather than
several subtly different ones.

This account exposes no rerank endpoint (verified against GET /v1/models), so
reranking has no adapter here; `core.reranking.base.Reranker` stays the seam
for one to be added.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.errors import ErrorCode, ProviderError
from core.reasoning.base import TokenUsage

PROVIDER_NAME = "mistral"

# mistral-embed emits fixed 1024-dimensional vectors; the width is not
# configurable and not reducible.
MISTRAL_EMBED_MODEL = "mistral-embed"
MISTRAL_EMBED_DIMENSIONS = 1024

# The API accepts up to 512 inputs per call.
MAX_BATCH_SIZE = 512

# Balanced default for grounded answering; overridden by GENERATION_MODEL.
MISTRAL_CHAT_MODEL = "mistral-medium-latest"

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


class MistralTransport:
    """HTTP plumbing shared by every Mistral capability."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._max_retries = max_retries
        # Held here and applied per request rather than baked into a client the
        # adapter may not own: an injected client (tests, a pooled or proxied
        # client) would otherwise send unauthenticated requests. The key still
        # appears in exactly one place and never at a call site.
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        model_id: str,
        default_code: ErrorCode,
    ) -> dict[str, Any]:
        """POST with exponential backoff on transient failures only.

        A malformed request or a bad key is not retried — repeating it just
        burns the rate limit and delays the real error.

        `default_code` is the caller's stage code, used for statuses with no
        specific meaning, so a failure stays attributable to the stage that
        caused it rather than collapsing to a generic internal error.
        """
        attempts = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        async for attempt in attempts:
            with attempt:
                return await self._post_once(
                    path, body, model_id=model_id, default_code=default_code
                )
        raise AssertionError("unreachable: reraise=True")  # pragma: no cover

    async def _post_once(
        self,
        path: str,
        body: dict[str, Any],
        *,
        model_id: str,
        default_code: ErrorCode,
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "The request to Mistral timed out.",
                code=ErrorCode.MODEL_TIMEOUT,
                provider=PROVIDER_NAME,
                model=model_id,
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Mistral could not be reached.",
                code=ErrorCode.MODEL_UNAVAILABLE,
                provider=PROVIDER_NAME,
                model=model_id,
                retryable=True,
            ) from exc

        if response.is_success:
            data: dict[str, Any] = response.json()
            return data

        raise self._error_for(response, model_id=model_id, default=default_code)

    def _error_for(
        self, response: httpx.Response, *, model_id: str, default: ErrorCode
    ) -> ProviderError:
        """Map an HTTP status onto the domain taxonomy.

        The response body is deliberately not attached: it can echo the request,
        and `details` is serialised into API responses.
        """
        status = response.status_code
        retryable = status in _RETRYABLE_STATUS

        if status in (401, 403):
            code = ErrorCode.PROVIDER_NOT_CONFIGURED
            message = "The Mistral API key was rejected."
        elif status == 429:
            code = ErrorCode.MODEL_RATE_LIMIT
            message = "The Mistral rate limit was exceeded."
        elif status == 422:
            code = ErrorCode.MODEL_CONTEXT_EXCEEDED
            message = "Mistral rejected the request payload."
        elif status >= 500:
            code = ErrorCode.MODEL_UNAVAILABLE
            message = "Mistral is unavailable."
        else:
            code = default
            message = f"Mistral returned an unexpected status ({status})."

        return ProviderError(
            message,
            code=code,
            provider=PROVIDER_NAME,
            model=model_id,
            retryable=retryable,
            details={"status_code": status},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class MistralEmbeddingModel:
    """Implements `EmbeddingModel` against POST /v1/embeddings."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.mistral.ai/v1",
        model_id: str = MISTRAL_EMBED_MODEL,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        transport: MistralTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self.dimensions = MISTRAL_EMBED_DIMENSIONS
        self.max_batch_size = MAX_BATCH_SIZE
        self._transport = transport or MistralTransport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > self.max_batch_size:
            raise ProviderError(
                f"Batch of {len(texts)} exceeds the provider limit of {self.max_batch_size}.",
                code=ErrorCode.MODEL_CONTEXT_EXCEEDED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            )

        payload = await self._transport.post(
            "/embeddings",
            {"model": self.model_id, "input": texts},
            model_id=self.model_id,
            default_code=ErrorCode.EMBEDDING_FAILED,
        )
        return self._parse_embeddings(payload, expected=len(texts))

    def _parse_embeddings(self, payload: dict[str, Any], *, expected: int) -> list[list[float]]:
        """Extract vectors, ordered to match the inputs.

        The response carries an explicit `index`, so ordering is restored from
        it rather than assumed — a positional mismatch would silently attach
        the wrong vector to a chunk.
        """
        try:
            items = payload["data"]
            ordered = sorted(items, key=lambda item: int(item["index"]))
            vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "The embedding response could not be parsed.",
                code=ErrorCode.EMBEDDING_FAILED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            ) from exc

        if len(vectors) != expected:
            raise ProviderError(
                f"Expected {expected} embeddings but received {len(vectors)}.",
                code=ErrorCode.EMBEDDING_FAILED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ProviderError(
                    f"Expected {self.dimensions}-dimensional vectors, received {len(vector)}.",
                    code=ErrorCode.EMBEDDING_FAILED,
                    provider=PROVIDER_NAME,
                    model=self.model_id,
                )
        return vectors

    async def aclose(self) -> None:
        await self._transport.aclose()


class MistralChatModel:
    """Implements `ChatModel` against POST /v1/chat/completions.

    Uses the provider's JSON mode so the response is structured by the API
    rather than by coaxing prose into shape and hoping (spec section 72:
    require structured output wherever practical).
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://api.mistral.ai/v1",
        model_id: str = MISTRAL_CHAT_MODEL,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        transport: MistralTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self._transport = transport or MistralTransport(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float,
    ) -> tuple[dict[str, Any], TokenUsage]:
        payload = await self._transport.post(
            "/chat/completions",
            {
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            model_id=self.model_id,
            default_code=ErrorCode.GENERATION_FAILED,
        )
        return self._parse_content(payload), self._parse_usage(payload)

    def _parse_content(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "The chat response had no message content.",
                code=ErrorCode.GENERATION_FAILED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            ) from exc

        # A truncated response is invalid JSON, and the cause (hitting the
        # output cap) is worth naming rather than reporting as a parse error.
        if choice.get("finish_reason") == "length":
            raise ProviderError(
                "The answer was cut off by the output token limit.",
                code=ErrorCode.MODEL_CONTEXT_EXCEEDED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "The chat response was not valid JSON.",
                code=ErrorCode.GENERATION_FAILED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            ) from exc

        if not isinstance(parsed, dict):
            raise ProviderError(
                "The chat response was not a JSON object.",
                code=ErrorCode.GENERATION_FAILED,
                provider=PROVIDER_NAME,
                model=self.model_id,
            )
        return parsed

    def _parse_usage(self, payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage") or {}
        return TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    async def aclose(self) -> None:
        await self._transport.aclose()
