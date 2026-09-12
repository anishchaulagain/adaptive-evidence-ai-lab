"""Keyword retrieval over Postgres full-text search (spec section 15B).

Deliberately *not* named BM25. Spec section 6 permits Postgres FTS as the
initial keyword backend, but `ts_rank_cd` is cover-density ranking, not
Okapi BM25: there is no tunable k1/b, and term saturation and document-length
normalisation behave differently. Calling it BM25 would quietly corrupt the
strategy comparisons this platform exists to run, so the name and the reported
`retriever` value say what actually ran. `KEYWORD_BACKEND=bm25` is reserved for
a true implementation behind this same protocol.

Lexical retrieval is what finds names, acronyms, identifiers, error codes and
numbers — the cases where a dense embedding of the query can drift to a
topically similar but factually wrong passage.

Two measured properties that hybrid fusion (Phase 5) must account for:

* **Conjunctive by default.** `websearch_to_tsquery` joins terms with AND, so
  a chunk must contain *every* query lexeme to match at all. This arm is
  therefore high-precision and low-recall: a natural-language question usually
  returns nothing, an exact identifier returns just the right passage. That
  complements dense retrieval rather than duplicating it, but it does mean the
  keyword arm often contributes no candidates.
* **No IDF.** `ts_rank_cd` scores term frequency and proximity only; a rare
  identifier and a common word carry the same weight. This is the concrete gap
  that a real BM25 backend would close, and the reason OR semantics were not
  adopted here — without IDF, widening to OR would rank common-word matches
  alongside exact ones.

Scores are comparable between chunks for one query, not across queries: cover
density falls as a query spans more terms, so a five-term match can score below
a one-term match.
"""

from __future__ import annotations

from sqlalchemy import cast, func, literal, select
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import FTS_LANGUAGE, Chunk
from core.retrieval.base import RetrievalQuery
from core.types import Provenance, RetrievedChunk, RetrieverKind

# ts_rank_cd normalisation flag 32 maps the raw rank to rank/(rank+1), giving a
# score in [0, 1) that is comparable in shape to the cosine similarity the
# dense retriever returns. Fusion in Phase 5 is rank-based, so the absolute
# scale matters less than both retrievers reporting *some* honest score.
_RANK_NORMALIZATION = 32


class PostgresFtsRetriever:
    """Implements `Retriever` against the generated `chunks.tsv` column."""

    name = "postgres_fts_keyword"

    def __init__(self, session: AsyncSession, *, config: str = FTS_LANGUAGE) -> None:
        self._session = session
        self._config = config

    def _regconfig(self) -> object:
        return cast(literal(self._config), REGCONFIG)

    async def query_lexemes(self, text: str) -> list[str]:
        """The lexemes a query reduces to, after stemming and stopword removal.

        Exposed because it explains a result: a search for `get_user_by_id`
        becomes `get & user & id`, and knowing that is the difference between
        a surprising ranking and an understood one.

        Ordered by first position rather than alphabetically — which is what
        `tsvector_to_array` would give — so the terms read in query order.
        """
        statement = sql_text(
            "SELECT array_agg(lexeme ORDER BY positions[1]) "
            "FROM unnest(to_tsvector(CAST(:fts_config AS regconfig), :fts_text))"
        ).bindparams(fts_config=self._config, fts_text=text)
        lexemes = await self._session.scalar(statement)
        return list(lexemes or [])

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Return the best lexical matches, with their matched terms."""
        # websearch_to_tsquery accepts anything a user might type — quoted
        # phrases, OR, leading minus — and never raises on malformed input,
        # unlike to_tsquery. A search box must not be able to 500 the API.
        tsquery = func.websearch_to_tsquery(self._regconfig(), query.text)
        rank = func.ts_rank_cd(Chunk.tsv, tsquery, _RANK_NORMALIZATION)

        statement = (
            select(Chunk, rank.label("score"), func.tsvector_to_array(Chunk.tsv))
            .where(
                Chunk.project_id == query.project_id,
                Chunk.tsv.op("@@")(tsquery),
            )
            .order_by(rank.desc(), Chunk.id)
            .limit(query.top_k)
        )
        rows = (await self._session.execute(statement)).all()
        if not rows:
            return []

        # Fetched once, not per row: the intersection is cheap in Python and
        # avoids shipping a raw-SQL array subquery through the ORM.
        wanted = await self.query_lexemes(query.text)

        results: list[RetrievedChunk] = []
        for rank_index, (chunk, score, lexemes) in enumerate(rows):
            present = set(lexemes or [])
            # Query order, not index order, so the terms read like the query.
            matched = tuple(term for term in wanted if term in present)
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=float(score),
                    retriever=RetrieverKind.KEYWORD,
                    provenance=Provenance(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        page=chunk.page,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                    ),
                    rank=rank_index,
                    matched_terms=matched,
                    metadata={"fts_config": self._config},
                )
            )
        return results
