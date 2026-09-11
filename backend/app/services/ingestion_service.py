"""Ingestion orchestration (spec section 10).

Enqueues work rather than doing it inline — an HTTP request must never block
on parsing, OCR or embedding (spec section 46).
"""

from __future__ import annotations

from app.services.base import Service


class IngestionService(Service):
    """Register documents, dispatch ingestion jobs and report status."""
