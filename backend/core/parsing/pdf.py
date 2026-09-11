"""PDF parsing.

Uses pypdf (BSD): pure Python, no system libraries, and permissively licensed
— PyMuPDF is AGPL and unsuitable for this project.

One block per page. Page-level granularity is what makes a citation
verifiable: a reader can open the document at that page and see the quote.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core.errors import ErrorCode, PipelineError
from core.parsing.base import ParsedBlock, ParsedDocument


class PdfParser:
    """Implements `Parser` for application/pdf."""

    mime_types = frozenset({"application/pdf"})

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(data))
        except (PdfReadError, ValueError, OSError) as exc:
            raise PipelineError(
                "The file could not be read as a PDF.",
                code=ErrorCode.PARSING_FAILED,
                stage="parsing",
                details={"filename": filename},
            ) from exc

        if reader.is_encrypted:
            # An empty password unlocks many PDFs that are merely permission
            # protected; a real password is a genuine failure.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise PipelineError(
                    "The PDF is password protected.",
                    code=ErrorCode.PARSING_FAILED,
                    stage="parsing",
                    details={"filename": filename},
                ) from exc

        blocks: list[ParsedBlock] = []
        offset = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:
                raise PipelineError(
                    f"Page {number} could not be extracted.",
                    code=ErrorCode.PARSING_FAILED,
                    stage="parsing",
                    details={"filename": filename, "page": number},
                ) from exc

            if not text:
                # Scanned pages yield nothing until OCR lands; skipping them
                # keeps offsets honest rather than inventing empty blocks.
                continue

            blocks.append(
                ParsedBlock(
                    text=text,
                    page=number,
                    char_start=offset,
                    char_end=offset + len(text),
                    kind="page",
                )
            )
            offset += len(text) + 1

        return ParsedDocument(
            blocks=tuple(blocks),
            page_count=len(reader.pages),
            metadata={"filename": filename} if filename else {},
        )
