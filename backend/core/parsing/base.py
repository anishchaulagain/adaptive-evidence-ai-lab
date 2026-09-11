"""Parser interface.

A parser turns a raw byte stream into structured text plus layout metadata.
One implementation per format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """A positioned unit of text: paragraph, heading, table cell, caption."""

    text: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    kind: str = "paragraph"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    blocks: tuple[ParsedBlock, ...]
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Parser(Protocol):
    """Implementations must preserve positions - citations depend on them."""

    mime_types: frozenset[str]

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument: ...
