"""Chunking invariants.

These properties are what make citations resolvable and re-ingestion stable,
so they are tested directly rather than through the ingestion endpoint.
"""

from __future__ import annotations

import pytest

from core.chunking.recursive import RecursiveCharacterChunker
from core.chunking.tokens import HeuristicTokenCounter
from core.parsing.base import ParsedBlock, ParsedDocument

pytestmark = pytest.mark.unit


def _document(text: str, *, page: int | None = 1) -> ParsedDocument:
    return ParsedDocument(
        blocks=(ParsedBlock(text=text, page=page, char_start=0, char_end=len(text)),),
        page_count=1,
    )


def _chunker(size: int = 100, overlap: int = 20) -> RecursiveCharacterChunker:
    return RecursiveCharacterChunker(chunk_size=size, overlap=overlap)


def test_short_text_is_a_single_chunk() -> None:
    chunks = _chunker().chunk(_document("A short paragraph."))

    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."
    assert chunks[0].ordinal == 0


def test_long_text_is_split_into_multiple_chunks() -> None:
    text = ". ".join(f"Sentence number {i}" for i in range(40))

    chunks = _chunker().chunk(_document(text))

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)


def test_chunking_is_deterministic() -> None:
    """Re-ingesting a document must not move chunk boundaries, or every
    existing citation silently changes meaning."""
    text = ". ".join(f"Sentence number {i}" for i in range(40))

    first = _chunker().chunk(_document(text))
    second = _chunker().chunk(_document(text))

    assert [(c.text, c.char_start, c.char_end) for c in first] == [
        (c.text, c.char_start, c.char_end) for c in second
    ]


def test_ordinals_are_contiguous_across_blocks() -> None:
    document = ParsedDocument(
        blocks=(
            ParsedBlock(text="First block.", page=1, char_start=0, char_end=12),
            ParsedBlock(text="Second block.", page=2, char_start=13, char_end=26),
        ),
        page_count=2,
    )

    chunks = _chunker().chunk(document)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_offsets_resolve_back_to_the_source_text() -> None:
    """The provenance contract: slicing the source by a chunk's offsets must
    return that chunk's text."""
    text = ". ".join(f"Sentence number {i}" for i in range(40))

    for chunk in _chunker().chunk(_document(text)):
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_page_is_carried_onto_every_chunk() -> None:
    text = ". ".join(f"Sentence number {i}" for i in range(40))

    chunks = _chunker().chunk(_document(text, page=7))

    assert {chunk.page for chunk in chunks} == {7}


def test_chunks_overlap_so_context_spans_boundaries() -> None:
    text = ". ".join(f"Sentence number {i}" for i in range(40))

    chunks = _chunker(size=100, overlap=30).chunk(_document(text))

    first_end, second_start = chunks[0].char_end, chunks[1].char_start
    assert first_end is not None and second_start is not None
    assert second_start < first_end


def test_a_word_longer_than_the_chunk_size_still_terminates() -> None:
    """A pathological input must not loop forever or drop content."""
    chunks = _chunker(size=50, overlap=10).chunk(_document("x" * 500))

    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks).count("x") >= 500


def test_whitespace_only_blocks_produce_no_chunks() -> None:
    assert _chunker().chunk(_document("   \n\n   ")) == []


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_configuration_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=size, overlap=overlap)


def test_token_count_is_a_positive_estimate() -> None:
    counter = HeuristicTokenCounter()

    assert counter.count("") == 0
    assert counter.count("a") == 1
    assert counter.count("word " * 100) > 50
