"""Adaptive query analyzer (spec sections 14, 76).

Deterministic rules, as §76 directs. Prefer ordinary logic where it suffices:
classifying a query by the shape of its tokens costs nothing, is reproducible,
and does not need an LLM (spec rule 20). A learned policy can replace this
behind the same protocol once there is evidence the rules are the limit.

Every classification records which rules fired, so a downstream retrieval
choice can be explained rather than asserted.
"""

from __future__ import annotations

import re
from typing import Protocol

from core.types import QueryAnalysis, QueryType

# An identifier-shaped token: ERR-5521, get_user_by_id, v4.2.1, SKU12345.
_IDENTIFIER = re.compile(r"\b(?=\S*[\d_\-/.])(?=\S*[A-Za-z])\S{3,}\b")
# An acronym: two or more capitals, not sentence-initial capitalisation.
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_QUOTED = re.compile(r'"([^"]{2,})"|\'([^\']{2,})\'')
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
# A version or decimal figure: 4.2.1, 1.5. It carries no letter, so the
# identifier pattern misses it — yet it is exactly the kind of token lexical
# retrieval pins down and a dense embedding blurs.
_VERSION_OR_DECIMAL = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")

_COMPARATIVE = re.compile(
    r"\b(vs\.?|versus|compare[ds]?|comparison|difference between|better than|"
    r"worse than|rather than)\b",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r"\b(before|after|since|until|during|when|latest|current|recent|"
    r"previously)\b",
    re.IGNORECASE,
)
_MULTI_HOP = re.compile(
    r"\b(and then|which also|both .+ and|as well as|in addition to|"
    r"followed by)\b",
    re.IGNORECASE,
)
_NUMERIC_REASONING = re.compile(
    r"\b(how many|how much|average|mean|median|total|sum|percent|percentage|"
    r"rate|ratio|more than|less than|at least|at most|between)\b",
    re.IGNORECASE,
)
_TECHNICAL = re.compile(
    r"\b(error|exception|stack ?trace|endpoint|api|schema|migration|config|"
    r"deploy|timeout|null|undefined|function|method|class|query|index)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"\b(what|why|how|when|where|who|which|explain|describe)\b", re.IGNORECASE)

# Below this many meaningful words, a query carries too little signal to
# classify confidently; it is marked ambiguous rather than guessed at.
_AMBIGUOUS_WORD_COUNT = 2


class QueryAnalyzer(Protocol):
    def analyze(self, text: str) -> QueryAnalysis: ...


class RuleBasedQueryAnalyzer:
    """Implements `QueryAnalyzer` with deterministic token rules."""

    name = "rule_based"

    def analyze(self, text: str) -> QueryAnalysis:
        stripped = text.strip()
        words = [word for word in re.split(r"\s+", stripped) if word]
        signals: list[str] = []

        entities = self._entities(stripped, signals)
        has_exact = bool(entities)

        non_year_numbers = [
            token for token in _NUMBER.findall(stripped) if not _YEAR.fullmatch(token)
        ]
        # Phrasing ("how many", "average") says the *task* is numeric; a bare
        # figure only says a number appears. They imply different profiles.
        numeric_reasoning = bool(_NUMERIC_REASONING.search(stripped))
        numeric = numeric_reasoning or bool(non_year_numbers)
        if numeric:
            signals.append("numeric_reasoning" if numeric_reasoning else "numeric_token")
        comparative = bool(_COMPARATIVE.search(stripped))
        if comparative:
            signals.append("comparative_phrase")
        temporal = bool(_TEMPORAL.search(stripped)) or bool(_YEAR.search(stripped))
        if temporal:
            signals.append("temporal_marker")
        multihop = bool(_MULTI_HOP.search(stripped)) or stripped.count("?") > 1
        if multihop:
            signals.append("multi_hop_phrase")
        technical = bool(_TECHNICAL.search(stripped))
        if technical:
            signals.append("technical_vocabulary")

        ambiguity = self._ambiguity(words, stripped, signals)
        query_type = self._classify(
            words=words,
            has_exact=has_exact,
            numeric=numeric,
            numeric_reasoning=numeric_reasoning,
            comparative=comparative,
            temporal=temporal,
            multihop=multihop,
            technical=technical,
            ambiguity=ambiguity,
        )
        signals.append(f"type:{query_type}")

        return QueryAnalysis(
            query_type=query_type,
            difficulty=self._difficulty(
                words=words,
                multihop=multihop,
                comparative=comparative,
                ambiguity=ambiguity,
            ),
            ambiguity=ambiguity,
            requires_exact_match=has_exact,
            requires_multihop=multihop,
            requires_numeric_reasoning=numeric_reasoning,
            entities=entities,
            signals=tuple(signals),
        )

    def _entities(self, text: str, signals: list[str]) -> tuple[str, ...]:
        """Tokens that must match literally.

        A quoted phrase is an explicit request for exactness; an acronym or an
        identifier is an implicit one. These are the cases where a dense
        embedding can land on a topically similar passage naming the wrong
        thing entirely.
        """
        found: list[str] = []
        for match in _QUOTED.finditer(text):
            found.append(match.group(1) or match.group(2))
        if found:
            signals.append("quoted_phrase")

        identifiers = [
            token
            for token in _IDENTIFIER.findall(text)
            # A bare year is temporal, not an identifier.
            if not _YEAR.fullmatch(token)
        ]
        if identifiers:
            signals.append("identifier_token")
            found.extend(identifiers)

        versions = _VERSION_OR_DECIMAL.findall(text)
        if versions:
            signals.append("version_or_decimal")
            found.extend(versions)

        acronyms = [
            token
            for token in _ACRONYM.findall(text)
            if not any(token in identifier for identifier in identifiers)
        ]
        if acronyms:
            signals.append("acronym")
            found.extend(acronyms)

        # Order-preserving deduplication, so the reason reads as written.
        seen: dict[str, None] = {}
        for item in found:
            seen.setdefault(item, None)
        return tuple(seen)

    def _ambiguity(self, words: list[str], text: str, signals: list[str]) -> float:
        if len(words) <= _AMBIGUOUS_WORD_COUNT:
            signals.append("very_short")
            return 0.8
        if not _QUESTION.search(text) and len(words) <= 4:
            signals.append("no_question_word")
            return 0.5
        return 0.1

    def _difficulty(
        self, *, words: list[str], multihop: bool, comparative: bool, ambiguity: float
    ) -> float:
        """A rough 0-1 estimate, not a measurement.

        Deliberately crude: it exists to order queries, and Phase 14's
        experiments are what will say whether it correlates with anything.
        """
        score = 0.3
        if multihop:
            score += 0.3
        if comparative:
            score += 0.15
        if len(words) > 15:
            score += 0.15
        score += ambiguity * 0.2
        return round(min(1.0, score), 3)

    def _classify(
        self,
        *,
        words: list[str],
        has_exact: bool,
        numeric: bool,
        numeric_reasoning: bool,
        comparative: bool,
        temporal: bool,
        multihop: bool,
        technical: bool,
        ambiguity: float,
    ) -> QueryType:
        """Assign one type, most specific first.

        The order is the policy: an identifier-bearing query is treated as
        exact even when it also looks technical, because the identifier is the
        part that lexical retrieval can pin down.
        """
        if not words:
            return QueryType.AMBIGUOUS
        if multihop:
            return QueryType.MULTI_HOP
        # Numeric *phrasing* outranks a bare identifier: "how many ERR-5521
        # events" is a counting task, not a lookup.
        if numeric_reasoning:
            return QueryType.NUMERIC
        if has_exact:
            return QueryType.EXACT_ENTITY
        if comparative:
            return QueryType.COMPARATIVE
        if numeric:
            return QueryType.NUMERIC
        if temporal:
            return QueryType.TEMPORAL
        if technical:
            return QueryType.TECHNICAL
        if ambiguity >= 0.8:
            return QueryType.AMBIGUOUS
        return QueryType.CONCEPTUAL
