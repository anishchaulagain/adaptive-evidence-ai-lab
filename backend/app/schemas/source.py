"""Data source schemas (spec section 9)."""

from __future__ import annotations

from app.schemas.common import APIModel, IdentifiedModel


class SourceCreate(APIModel):
    """Fields: name, kind, connection_config.

    `connection_config` may contain credentials — they are encrypted at rest
    and never echoed back in `SourceRead`.
    """


class SourceRead(IdentifiedModel):
    """Fields: name, kind, status, last_synced_at, document_count. No secrets."""
