"""Explicit error taxonomy.

Every pipeline stage fails with a named code (spec section 47) so that failures
are attributable in traces and in failure analysis, rather than surfacing as an
anonymous 500.
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
    STORAGE_FAILED = "STORAGE_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base class for every error the application raises deliberately.

    `details` must never carry secrets — it is serialised into API responses.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}


# --- transport-shaped errors ---------------------------------------------


class NotFoundError(AppError):
    code = ErrorCode.NOT_FOUND
    status_code = 404


class ConflictError(AppError):
    code = ErrorCode.CONFLICT
    status_code = 409


class ValidationError(AppError):
    code = ErrorCode.VALIDATION_ERROR
    status_code = 422


class UnauthenticatedError(AppError):
    code = ErrorCode.UNAUTHENTICATED
    status_code = 401


class ForbiddenError(AppError):
    code = ErrorCode.FORBIDDEN
    status_code = 403


class RateLimitedError(AppError):
    code = ErrorCode.RATE_LIMITED
    status_code = 429


# --- pipeline-shaped errors ----------------------------------------------


class PipelineError(AppError):
    """Raised by a stage in the AI pipeline. Always recorded on the trace."""

    status_code = 500

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


class ProviderError(AppError):
    """Raised by a model provider adapter. Never include the API key."""

    status_code = 502

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
