"""Logging safety properties."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.logging import configure_logging, get_logger

pytestmark = pytest.mark.unit


def test_secrets_are_redacted(settings: Settings) -> None:
    """A credential must never reach the log sink, whatever the call site does."""
    from app.core.logging import _redact_secrets

    event = _redact_secrets(
        None, "info", {"api_key": "sk-real-secret", "database_url": "postgres://u:p@h/d"}
    )

    assert event["api_key"] == "[redacted]"
    assert event["database_url"] == "[redacted]"


def test_non_sensitive_fields_are_preserved() -> None:
    from app.core.logging import _redact_secrets

    event = _redact_secrets(None, "info", {"document_id": "abc", "chunk_count": 3})

    assert event == {"document_id": "abc", "chunk_count": 3}


def test_non_ascii_text_does_not_break_logging(settings: Settings) -> None:
    """Document text is logged; a legacy console codepage must not turn that
    into a logging failure."""
    configure_logging(settings)
    logger = get_logger("test")

    logger.info("ingestion.completed", title="气候报告 — résumé → summary")
