"""Error taxonomy invariants (spec section 47)."""

from __future__ import annotations

import pytest

from app.core.errors import (
    AppError,
    ConflictError,
    DomainError,
    ErrorCode,
    NotFoundError,
    PipelineError,
    ProviderError,
    status_code_for,
)

pytestmark = pytest.mark.unit


def test_error_codes_are_stable_strings() -> None:
    """Codes are persisted in traces, so their values must not drift."""
    assert ErrorCode.RETRIEVAL_FAILED == "RETRIEVAL_FAILED"
    assert str(ErrorCode.MODEL_RATE_LIMIT) == "MODEL_RATE_LIMIT"


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (NotFoundError("missing"), 404, ErrorCode.NOT_FOUND),
        (ConflictError("duplicate"), 409, ErrorCode.CONFLICT),
    ],
)
def test_transport_errors_carry_status_and_code(
    error: AppError, status_code: int, code: ErrorCode
) -> None:
    assert error.status_code == status_code
    assert error.code is code


def test_pipeline_error_records_its_stage() -> None:
    """Failure attribution depends on knowing which stage failed."""
    error = PipelineError(
        "embedding provider timed out",
        code=ErrorCode.EMBEDDING_FAILED,
        stage="embedding",
    )
    assert error.stage == "embedding"
    assert error.code is ErrorCode.EMBEDDING_FAILED


def test_provider_error_marks_retryability() -> None:
    error = ProviderError(
        "rate limited",
        code=ErrorCode.MODEL_RATE_LIMIT,
        provider="anthropic",
        retryable=True,
    )
    assert error.retryable is True
    assert error.provider == "anthropic"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderError("slow", code=ErrorCode.MODEL_TIMEOUT, provider="p"), 504),
        (ProviderError("429", code=ErrorCode.MODEL_RATE_LIMIT, provider="p"), 429),
        (PipelineError("boom", code=ErrorCode.PARSING_FAILED, stage="parsing"), 500),
        (NotFoundError("missing"), 404),
    ],
)
def test_core_errors_map_to_honest_statuses(error: DomainError, expected: int) -> None:
    """An error raised deep in the pipeline still gets a meaningful status."""
    assert status_code_for(error) == expected


def test_details_default_to_empty_rather_than_none() -> None:
    """`details` is serialised into responses; it must always be a mapping."""
    assert AppError("boom").details == {}
