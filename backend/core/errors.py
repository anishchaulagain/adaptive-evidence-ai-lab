"""Domain error taxonomy.

Lives in `core` because error codes are domain concepts: a pipeline stage that
fails must be attributable in a trace whether it ran inside a request, a
worker, or an experiment script. The HTTP status mapping is added by
`app.core.errors`, which is the only layer that knows about HTTP.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable, machine-readable failure codes. Values appear in traces and logs."""

    # --- pipeline stages (spec section 47) ---
    INGESTION_FAILED = "INGESTION_FAILED"
    PARSING_FAILED = "PARSING_FAILED"
    CHUNKING_FAILED = "CHUNKING_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    INDEXING_FAILED = "INDEXING_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    RERANKING_FAILED = "RERANKING_FAILED"
    FUSION_FAILED = "FUSION_FAILED"
    ROUTING_FAILED = "ROUTING_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    EXPERIMENT_FAILED = "EXPERIMENT_FAILED"
    BENCHMARK_FAILED = "BENCHMARK_FAILED"

    # --- model providers ---
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_CONTEXT_EXCEEDED = "MODEL_CONTEXT_EXCEEDED"
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"

    # --- transport / application ---
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    STORAGE_FAILED = "STORAGE_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DomainError(Exception):
    """Base class for every error the platform raises deliberately.

    `details` must never carry secrets — it is serialised into API responses.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}


class PipelineError(DomainError):
    """Raised by a stage in the AI pipeline. Always recorded on the trace."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.stage = stage


class ProviderError(DomainError):
    """Raised by a model provider adapter. Never include the API key."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        provider: str,
        model: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.provider = provider
        self.model = model
        self.retryable = retryable


class StorageError(DomainError):
    """Object storage failed."""

    code = ErrorCode.STORAGE_FAILED
