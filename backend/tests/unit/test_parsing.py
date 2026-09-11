"""Parser behaviour, including real PDF bytes.

PDFs are generated rather than committed as fixtures, so the tests stay
readable and there is no binary blob to trust.
"""

from __future__ import annotations

import io

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from core.errors import ErrorCode, PipelineError
from core.parsing.pdf import PdfParser
from core.parsing.registry import get_parser, supported_mime_types
from core.parsing.text import TextParser

pytestmark = pytest.mark.unit


def _pdf(pages: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for text in pages:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


# --- registry -------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime_type", "expected"),
    [
        ("application/pdf", PdfParser),
        ("text/plain", TextParser),
        ("text/markdown", TextParser),
    ],
)
def test_registry_selects_the_right_parser(mime_type: str, expected: type) -> None:
    assert isinstance(get_parser(mime_type), expected)


def test_unsupported_type_names_what_is_supported() -> None:
    with pytest.raises(PipelineError) as excinfo:
        get_parser("application/vnd.ms-excel")

    assert excinfo.value.code is ErrorCode.PARSING_FAILED
    assert excinfo.value.stage == "parsing"
    assert "application/pdf" in excinfo.value.details["supported"]


def test_supported_mime_types_covers_the_phase_2_formats() -> None:
    assert supported_mime_types() == {"application/pdf", "text/plain", "text/markdown"}


# --- pdf ------------------------------------------------------------------


def test_pdf_yields_one_block_per_page_with_page_numbers() -> None:
    document = PdfParser().parse(_pdf(["First page text", "Second page text"]))

    assert document.page_count == 2
    assert [block.page for block in document.blocks] == [1, 2]
    assert "First page text" in document.blocks[0].text


def test_pdf_blocks_carry_monotonic_offsets() -> None:
    document = PdfParser().parse(_pdf(["Alpha", "Beta", "Gamma"]))

    for block in document.blocks:
        assert block.char_start is not None
        assert block.char_end is not None
        assert block.char_end - block.char_start == len(block.text)

    starts = [block.char_start for block in document.blocks if block.char_start is not None]
    assert starts == sorted(starts)


def test_pdf_without_extractable_text_yields_no_blocks() -> None:
    """A scanned page produces nothing until OCR lands — it must not invent
    empty blocks that would corrupt offsets."""
    document = PdfParser().parse(_pdf([" "]))

    assert document.blocks == ()
    assert document.page_count == 1


def test_unreadable_bytes_fail_with_parsing_failed() -> None:
    with pytest.raises(PipelineError) as excinfo:
        PdfParser().parse(b"this is definitely not a pdf")

    assert excinfo.value.code is ErrorCode.PARSING_FAILED


# --- text and markdown ----------------------------------------------------


def test_text_is_split_on_blank_lines() -> None:
    document = TextParser().parse(b"First para.\n\nSecond para.")

    assert [block.text for block in document.blocks] == ["First para.", "Second para."]


def test_text_offsets_resolve_back_to_the_source() -> None:
    raw = "First para.\n\n  Indented para.  \n\nThird."
    document = TextParser().parse(raw.encode())

    for block in document.blocks:
        assert raw[block.char_start : block.char_end] == block.text


def test_markdown_headings_are_labelled() -> None:
    document = TextParser().parse(b"# Title\n\nBody text.")

    assert document.blocks[0].kind == "heading"
    assert document.blocks[1].kind == "paragraph"


def test_invalid_utf8_does_not_fail_ingestion() -> None:
    document = TextParser().parse(b"caf\xe9 text")

    assert document.blocks[0].text.startswith("caf")
