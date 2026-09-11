"""Parser selection by MIME type."""

from __future__ import annotations

from core.errors import ErrorCode, PipelineError
from core.parsing.base import Parser
from core.parsing.pdf import PdfParser
from core.parsing.text import TextParser

# Phase 2 covers the formats the spec prioritises. DOCX and PPTX register here
# without any other module changing.
_PARSERS: tuple[Parser, ...] = (PdfParser(), TextParser())


def supported_mime_types() -> frozenset[str]:
    return frozenset().union(*(parser.mime_types for parser in _PARSERS))


def get_parser(mime_type: str) -> Parser:
    for parser in _PARSERS:
        if mime_type in parser.mime_types:
            return parser
    raise PipelineError(
        f"No parser is registered for {mime_type!r}.",
        code=ErrorCode.PARSING_FAILED,
        stage="parsing",
        details={"mime_type": mime_type, "supported": sorted(supported_mime_types())},
    )
