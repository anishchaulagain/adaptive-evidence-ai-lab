"""Settings validation — the guard that keeps unsafe config out of production."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.core.config import (
    AuthMode,
    Environment,
    LogFormat,
    Settings,
    validate_runtime_settings,
)

pytestmark = pytest.mark.unit


def _settings_with(settings: Settings, **overrides: object) -> Settings:
    """Build settings through full validation, so `mode="before"` validators run.

    `model_copy` would bypass them, which is exactly what these tests exercise.
    """
    return Settings.model_validate(
        {
            "DATABASE_URL": str(settings.DATABASE_URL),
            "REDIS_URL": str(settings.REDIS_URL),
            **overrides,
        }
    )


def _production(settings: Settings, **overrides: object) -> Settings:
    safe: dict[str, object] = {
        "ENVIRONMENT": Environment.PRODUCTION,
        "DEBUG": False,
        "LOG_FORMAT": LogFormat.JSON,
        "AUTH_MODE": AuthMode.JWT,
        "SECRET_KEY": SecretStr("a-real-secret"),
        "MISTRAL_API_KEY": SecretStr("a-real-key"),
    }
    return settings.model_copy(update={**safe, **overrides})


def test_development_settings_are_not_validated(settings: Settings) -> None:
    validate_runtime_settings(settings.model_copy(update={"DEBUG": True}))


def test_safe_production_settings_pass(settings: Settings) -> None:
    validate_runtime_settings(_production(settings))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"SECRET_KEY": SecretStr("change-me-is-the-template")}, "SECRET_KEY"),
        ({"DEBUG": True}, "DEBUG"),
        ({"LOG_FORMAT": LogFormat.CONSOLE}, "LOG_FORMAT"),
        ({"AUTH_MODE": AuthMode.DISABLED}, "AUTH_MODE"),
        ({"MISTRAL_API_KEY": None}, "MISTRAL_API_KEY"),
    ],
)
def test_unsafe_production_settings_are_rejected(
    settings: Settings, overrides: dict[str, object], expected: str
) -> None:
    with pytest.raises(RuntimeError, match=expected):
        validate_runtime_settings(_production(settings, **overrides))


def test_every_unsafe_value_is_reported_at_once(settings: Settings) -> None:
    """One boot attempt should reveal every problem, not just the first."""
    unsafe = _production(
        settings,
        DEBUG=True,
        LOG_FORMAT=LogFormat.CONSOLE,
        AUTH_MODE=AuthMode.DISABLED,
    )
    with pytest.raises(RuntimeError) as excinfo:
        validate_runtime_settings(unsafe)

    message = str(excinfo.value)
    assert "DEBUG" in message
    assert "LOG_FORMAT" in message
    assert "AUTH_MODE" in message


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_provider_key_is_treated_as_unset(settings: Settings, blank: str) -> None:
    """`.env` carries placeholder lines like `MISTRAL_API_KEY=`. A blank value
    must read as absent, not as a configured key that fails with a 401 later."""
    parsed = _settings_with(settings, MISTRAL_API_KEY=blank)

    assert parsed.MISTRAL_API_KEY is None


def test_a_real_provider_key_is_preserved(settings: Settings) -> None:
    parsed = _settings_with(settings, MISTRAL_API_KEY="sk-real-key")

    assert parsed.MISTRAL_API_KEY is not None
    assert parsed.MISTRAL_API_KEY.get_secret_value() == "sk-real-key"


def test_embedding_dimensions_must_match_the_column(settings: Settings) -> None:
    """A mismatch would write vectors the pgvector column cannot hold, failing
    deep inside a worker instead of at boot."""
    with pytest.raises(RuntimeError, match="EMBEDDING_DIMENSIONS"):
        validate_runtime_settings(settings.model_copy(update={"EMBEDDING_DIMENSIONS": 1536}))
