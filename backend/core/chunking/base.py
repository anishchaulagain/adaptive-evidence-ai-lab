"""Chunking interface.

Chunking is deterministic (spec rule 19) so re-ingesting the same document
yields identical chunk boundaries and stable citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.parsing.base import ParsedDocument


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    ordinal: int
    token_count: int
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    name: str

    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...
