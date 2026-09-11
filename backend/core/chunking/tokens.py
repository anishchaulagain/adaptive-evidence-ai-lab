"""Token counting.

Phase 2 has no embedding model yet, so token counts are an explicit estimate
rather than a pretend-exact number. Phase 3 registers the embedding model's
real tokenizer against the same protocol; nothing else changes.
"""

from __future__ import annotations

from typing import Protocol

# Empirically ~4 characters per token for English prose across common BPE
# vocabularies. Good enough to size chunks, not good enough to bill against.
_CHARS_PER_TOKEN = 4


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Character-based estimate. Deterministic and offline."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / _CHARS_PER_TOKEN))
