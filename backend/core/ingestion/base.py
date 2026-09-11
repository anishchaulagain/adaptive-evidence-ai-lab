"""Source connector interface (spec section 9)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A document discovered at a source, before parsing."""

    external_id: str
    title: str
    uri: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceConnector(Protocol):
    """Connectors only discover and fetch; they never parse or embed."""

    kind: str

    def discover(self) -> AsyncIterator[SourceDocument]: ...

    async def fetch(self, document: SourceDocument) -> bytes: ...
