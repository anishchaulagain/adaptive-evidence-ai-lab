"""Recursive character chunking.

Deterministic by construction (spec rule 19): the same document always yields
the same boundaries, so chunk IDs and citations stay stable across
re-ingestion.

Splits on the largest natural boundary that fits — paragraph, then sentence,
then word — and only cuts mid-word when a single word exceeds the chunk size.
Every chunk keeps the page and character offsets of the block it came from, so
a citation resolves to an exact span.
"""

from __future__ import annotations

import re

from core.chunking.base import Chunk
from core.chunking.tokens import HeuristicTokenCounter, TokenCounter
from core.parsing.base import ParsedBlock, ParsedDocument

# Ordered widest to narrowest.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
_WHITESPACE = re.compile(r"\s+")


class RecursiveCharacterChunker:
    """Implements `Chunker`."""

    name = "recursive_character"

    def __init__(
        self,
        *,
        chunk_size: int,
        overlap: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokens = token_counter or HeuristicTokenCounter()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block in document.blocks:
            for text, start in self._split_block(block):
                chunks.append(
                    Chunk(
                        text=text,
                        ordinal=len(chunks),
                        token_count=self._tokens.count(text),
                        page=block.page,
                        char_start=start,
                        char_end=start + len(text),
                    )
                )
        return chunks

    def _split_block(self, block: ParsedBlock) -> list[tuple[str, int]]:
        """Return (text, absolute_char_start) pairs for one block."""
        base = block.char_start or 0
        text = block.text
        if len(text) <= self._chunk_size:
            stripped = text.strip()
            if not stripped:
                return []
            lead = len(text) - len(text.lstrip())
            return [(stripped, base + lead)]

        pieces: list[tuple[str, int]] = []
        cursor = 0
        while cursor < len(text):
            end = min(cursor + self._chunk_size, len(text))
            if end < len(text):
                end = self._boundary(text, cursor, end)

            piece = text[cursor:end]
            stripped = piece.strip()
            if stripped:
                # Preserve the offset of the stripped text, not the raw slice.
                lead = len(piece) - len(piece.lstrip())
                pieces.append((stripped, base + cursor + lead))

            if end >= len(text):
                break
            # Step back by the overlap so context spans the boundary, while
            # always advancing to guarantee termination.
            cursor = max(cursor + 1, end - self._overlap)

        return pieces

    def _boundary(self, text: str, start: int, end: int) -> int:
        """Find the widest natural separator inside the window."""
        window = text[start:end]
        for separator in _SEPARATORS:
            index = window.rfind(separator)
            # Ignore a separator so early that the chunk would be mostly empty.
            if index > self._chunk_size // 4:
                return start + index + len(separator)
        return end
