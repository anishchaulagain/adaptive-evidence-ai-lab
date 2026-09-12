"""Keyword retrieval (spec section 69, Phase 4).

The corpus is built so that each probe has exactly one correct answer and at
least one topically similar distractor. That is what makes these tests about
lexical precision — the cases where a dense retriever drifts to a passage that
is *about* the right subject but names the wrong identifier.

Runs against real Postgres, real generated tsvectors and the real GIN index.
No provider key is involved: keyword retrieval uses none.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.config import Settings
from app.db.session import get_sessionmaker
from app.models.chunk import Chunk
from app.retrieval.postgres_fts import PostgresFtsRetriever
from core.retrieval.base import RetrievalQuery
from core.types import RetrieverKind

pytestmark = pytest.mark.integration

SEARCH = "/api/v1/search"

# One passage per line; blank lines separate chunks.
CORPUS = "\n\n".join(
    [
        "The NASA GISTEMP dataset records surface temperature anomalies.",
        "The NOAA MLOST dataset also records surface temperature anomalies.",
        "Deployment halted with error code ERR-5521 during the rollout.",
        "Deployment halted with error code ERR-9310 during the rollback.",
        "The service requires version 4.2.1 or later to start correctly.",
        "The service requires version 3.9.7 for legacy compatibility.",
        "Warming should be limited to 1.5 degrees above preindustrial levels.",
        "Warming of 2.7 degrees is projected under current policies.",
        "Call get_user_by_id before touching the session cache.",
        "Anthropogenic aerosols exert a net negative radiative forcing.",
        "Rapporteur Mbeki delivered the closing statement to the assembly.",
    ]
).encode()


@pytest.fixture
async def corpus(client: AsyncClient, settings: Settings) -> AsyncIterator[UUID]:
    """A project whose documents are ingested and chunked, but not embedded.

    Keyword retrieval must work with no vectors present at all.
    """
    created = await client.post("/api/v1/projects", json={"name": f"Keyword {uuid4().hex[:8]}"})
    project_id = UUID(created.json()["id"])

    from app.workers.tasks.ingestion import ingest_document

    response = await client.post(
        "/api/v1/documents/upload",
        params={"project_id": str(project_id)},
        files={"file": ("lexical.txt", CORPUS, "text/plain")},
    )
    await ingest_document({"settings": settings}, UUID(response.json()["id"]))

    yield project_id

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await session.execute(text("DELETE FROM projects WHERE id = :id"), {"id": str(project_id)})
        await session.commit()


async def _search(
    client: AsyncClient, project_id: UUID, query: str, **kwargs: object
) -> dict[str, Any]:
    response = await client.post(
        SEARCH,
        json={
            "project_id": str(project_id),
            "query": query,
            "strategy": "keyword",
            **kwargs,
        },
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# --- the lexical categories the spec names --------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("GISTEMP", "GISTEMP", id="acronym"),
        pytest.param("ERR-5521", "ERR-5521", id="error-code"),
        pytest.param("version 4.2.1", "4.2.1", id="version-identifier"),
        pytest.param("1.5 degrees", "1.5 degrees", id="number"),
        pytest.param("anthropogenic aerosols", "aerosols", id="rare-terminology"),
        pytest.param("Mbeki", "Mbeki", id="exact-name"),
    ],
)
async def test_lexical_queries_retrieve_the_exact_passage(
    client: AsyncClient, corpus: UUID, query: str, expected: str
) -> None:
    """Each probe has a near-identical distractor differing only in the
    identifier, so ranking the right one is a real discrimination."""
    body = await _search(client, corpus, query)

    results = body["results"]
    assert results, f"no hits for {query!r}"
    assert expected.lower() in results[0]["text"].lower()


async def test_the_distractor_is_not_returned_for_a_precise_identifier(
    client: AsyncClient, corpus: UUID
) -> None:
    """ERR-9310 shares every word with ERR-5521's passage except the code, so
    only the requested code may come back."""
    body = await _search(client, corpus, "ERR-5521")

    texts = [item["text"] for item in body["results"]]
    assert any("ERR-5521" in item for item in texts)
    assert not any("ERR-9310" in item for item in texts)


# --- what the response must expose (spec section 69) ----------------------


async def test_results_expose_score_rank_and_matched_terms(
    client: AsyncClient, corpus: UUID
) -> None:
    body = await _search(client, corpus, "GISTEMP dataset")

    assert body["strategy"] == "keyword"
    # No embedding ran, so no model may be claimed.
    assert body["embedding_model"] is None
    assert body["query_terms"]

    results = body["results"]
    assert [item["rank"] for item in results] == list(range(len(results)))
    assert results == sorted(results, key=lambda item: -item["score"])
    for item in results:
        assert 0.0 <= item["score"] < 1.0
        assert item["retriever"] == "keyword"
        assert item["matched_terms"], "a lexical hit must say what matched"


async def test_matched_terms_are_the_stemmed_query_terms_present(
    client: AsyncClient, corpus: UUID
) -> None:
    """Matched terms explain the hit, so they must be the lexemes actually
    shared with the query — not the raw words typed."""
    body = await _search(client, corpus, "anthropogenic aerosols")

    top = body["results"][0]
    assert set(top["matched_terms"]) <= set(body["query_terms"])
    assert "aerosol" in top["matched_terms"]


async def test_query_terms_reveal_stemming_and_stopword_removal(
    client: AsyncClient, corpus: UUID
) -> None:
    """`the` and `to` are stopwords and `limited` stems; surfacing the lexemes
    is what turns a surprising ranking into an explicable one."""
    body = await _search(client, corpus, "the warming limited to degrees")

    terms = body["query_terms"]
    assert "the" not in terms
    assert "to" not in terms
    assert "limit" in terms
    assert "warm" in terms


async def test_snake_case_identifiers_are_split_into_parts(
    client: AsyncClient, corpus: UUID
) -> None:
    """Documented limitation, not an aspiration: the Postgres parser splits
    `get_user_by_id` into `get`, `user`, `id` and drops `by` as a stopword. The
    passage is still retrievable, but not as an exact identifier — a true BM25
    backend or a `simple`-config column would be needed for that.
    """
    body = await _search(client, corpus, "get_user_by_id")

    assert body["query_terms"] == ["get", "user", "id"]
    assert "get_user_by_id" in body["results"][0]["text"]


# --- behaviour ------------------------------------------------------------


async def test_keyword_search_works_without_any_embeddings(
    client: AsyncClient, corpus: UUID
) -> None:
    """The corpus fixture never embeds, so this also proves keyword retrieval
    needs no provider key."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        embedded = list(
            await session.scalars(select(Chunk.embedding).where(Chunk.project_id == corpus))
        )
    assert all(vector is None for vector in embedded)

    body = await _search(client, corpus, "GISTEMP")

    assert body["results"]


async def test_a_query_matching_nothing_returns_no_hits(client: AsyncClient, corpus: UUID) -> None:
    """Lexical retrieval returns nothing rather than the least-bad match — the
    property that makes it complementary to dense retrieval."""
    body = await _search(client, corpus, "zzzzqqq nonexistent term")

    assert body["results"] == []


async def test_top_k_limits_the_result_count(client: AsyncClient, corpus: UUID) -> None:
    body = await _search(client, corpus, "dataset OR version OR degrees", top_k=2)

    assert len(body["results"]) <= 2


@pytest.mark.parametrize(
    "query",
    ["'unbalanced", "a & b | c", "!!!", "-", '"unclosed phrase'],
    ids=["quote", "operators", "punctuation", "bare-minus", "unclosed-phrase"],
)
async def test_malformed_queries_do_not_error(
    client: AsyncClient, corpus: UUID, query: str
) -> None:
    """`websearch_to_tsquery` tolerates anything a user types; `to_tsquery`
    would raise and turn a search box into a 500."""
    body = await _search(client, corpus, query)

    assert isinstance(body["results"], list)


async def test_results_carry_provenance(client: AsyncClient, corpus: UUID) -> None:
    body = await _search(client, corpus, "GISTEMP")

    for item in body["results"]:
        assert item["chunk_id"]
        assert item["document_id"]
        assert item["char_start"] is not None
        assert item["char_end"] is not None


async def test_retrieval_is_scoped_to_the_project(client: AsyncClient, corpus: UUID) -> None:
    other = await client.post("/api/v1/projects", json={"name": f"Empty {uuid4().hex[:8]}"})

    body = await _search(client, UUID(other.json()["id"]), "GISTEMP")

    assert body["results"] == []


# --- retriever unit-level contract ----------------------------------------


async def test_retriever_reports_its_kind_and_config(corpus: UUID) -> None:
    """The reported retriever must name what actually ran, so strategy
    comparisons in later phases cannot silently attribute the wrong one."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        hits = await PostgresFtsRetriever(session).retrieve(
            RetrievalQuery(text="GISTEMP", project_id=corpus, top_k=5)
        )

    assert hits
    assert all(hit.retriever is RetrieverKind.KEYWORD for hit in hits)
    assert hits[0].metadata["fts_config"] == "english"


async def test_keyword_matching_is_conjunctive(client: AsyncClient, corpus: UUID) -> None:
    """Measured property, not an assumption: every query lexeme must be present.

    This makes the keyword arm high-precision and low-recall, and means it
    often contributes no candidates to a natural-language question. Phase 5
    fusion has to tolerate an empty keyword result set.
    """
    present = await _search(client, corpus, "GISTEMP dataset")
    assert present["results"]

    # Same query plus one lexeme that appears nowhere alongside the others.
    absent = await _search(client, corpus, "GISTEMP dataset rollback")
    assert absent["results"] == []


async def test_scores_are_not_comparable_across_queries(client: AsyncClient, corpus: UUID) -> None:
    """Cover density falls as a query spans more terms, so a broader match can
    score below a narrower one. Scores rank within a query; they are not a
    global relevance measure, and must not be averaged or thresholded as one.
    """
    narrow = await _search(client, corpus, "GISTEMP")
    broad = await _search(client, corpus, "NASA GISTEMP dataset records anomalies")

    assert narrow["results"] and broad["results"]
    # Both find the same passage...
    assert "GISTEMP" in narrow["results"][0]["text"]
    assert "GISTEMP" in broad["results"][0]["text"]
    # ...yet the match on more terms scores lower.
    assert broad["results"][0]["score"] < narrow["results"][0]["score"]
