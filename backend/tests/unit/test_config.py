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


def _production(settings: Settings, **overrides: object) -> Settings:
    safe: dict[str, object] = {
        "ENVIRONMENT": Environment.PRODUCTION,
        "DEBUG": False,
        "LOG_FORMAT": LogFormat.JSON,
        "AUTH_MODE": AuthMode.JWT,
        "SECRET_KEY": SecretStr("a-real-secret"),
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
