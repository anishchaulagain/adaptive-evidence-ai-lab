"""Plain text and Markdown parsing.

Both are already text; the work is splitting into blocks while recording the
character offsets that citations are resolved against.
"""

from __future__ import annotations

import re

from core.parsing.base import ParsedBlock, ParsedDocument

# A blank line (optionally carrying whitespace) separates blocks.
_BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")


class TextParser:
    """Implements `Parser` for text/plain and text/markdown."""

    mime_types = frozenset({"text/plain", "text/markdown"})

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        # Decode leniently: a single bad byte should not fail an ingestion.
        text = data.decode("utf-8", errors="replace")

        blocks: list[ParsedBlock] = []
        cursor = 0
        for raw in _BLOCK_SEPARATOR.split(text):
            start = text.find(raw, cursor)
            if start == -1:  # pragma: no cover - defensive
                start = cursor
            cursor = start + len(raw)

            stripped = raw.strip()
            if not stripped:
                continue

            # Offsets must point at the stripped text, or a citation would
            # include the surrounding whitespace.
            offset = start + (len(raw) - len(raw.lstrip()))
            blocks.append(
                ParsedBlock(
                    text=stripped,
                    page=None,
                    char_start=offset,
                    char_end=offset + len(stripped),
                    kind="heading" if _MARKDOWN_HEADING.match(stripped) else "paragraph",
                )
            )

        return ParsedDocument(
            blocks=tuple(blocks),
            page_count=None,
            metadata={"filename": filename} if filename else {},
        )
