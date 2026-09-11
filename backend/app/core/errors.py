"""HTTP-shaped errors.

The taxonomy itself lives in `core.errors`; this module adds the one thing
that is genuinely transport-specific — the status code — and re-exports the
domain errors so application code has a single import site.
"""

from __future__ import annotations

from typing import Any

from core.errors import (
    DomainError,
    ErrorCode,
    PipelineError,
    ProviderError,
    StorageError,
)

# Domain errors carry no status code of their own. Codes raised deep in the
# pipeline still deserve an honest status, so map the ones with a clear HTTP
# meaning; anything unmapped is a server fault.
DEFAULT_STATUS_CODE = 500

_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.MODEL_RATE_LIMIT: 429,
    ErrorCode.MODEL_TIMEOUT: 504,
    ErrorCode.MODEL_UNAVAILABLE: 502,
    ErrorCode.MODEL_CONTEXT_EXCEEDED: 422,
    ErrorCode.PROVIDER_NOT_CONFIGURED: 503,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.RATE_LIMITED: 429,
}


class AppError(DomainError):
    """A domain error with an explicit HTTP status."""

    status_code: int = DEFAULT_STATUS_CODE

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        if status_code is not None:
            self.status_code = status_code


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


class UnsupportedMediaTypeError(AppError):
    code = ErrorCode.UNSUPPORTED_MEDIA_TYPE
    status_code = 415


class PayloadTooLargeError(AppError):
    code = ErrorCode.PAYLOAD_TOO_LARGE
    status_code = 413


def status_code_for(error: DomainError) -> int:
    """Status for any domain error, including ones raised deep in `core`.

    An explicit `status_code` on the error wins; otherwise the code decides.
    """
    explicit = getattr(error, "status_code", None)
    if explicit is not None:
        return int(explicit)
    return _STATUS_BY_CODE.get(error.code, DEFAULT_STATUS_CODE)


__all__ = [
    "AppError",
    "ConflictError",
    "DomainError",
    "ErrorCode",
    "ForbiddenError",
    "NotFoundError",
    "PayloadTooLargeError",
    "PipelineError",
    "ProviderError",
    "RateLimitedError",
    "StorageError",
    "UnauthenticatedError",
    "UnsupportedMediaTypeError",
    "ValidationError",
    "status_code_for",
]
