"""Dense retrieval over pgvector (spec section 15A).

Similarity is cosine distance against the HNSW index on `chunks.embedding`.
Scores are returned, not hidden: comparing retrieval strategies later (Phases
4-5) depends on every retriever exposing a comparable score and saying which
strategy produced each hit.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from core.embeddings.base import EmbeddingPipeline
from core.errors import ErrorCode, ProviderError
from core.retrieval.base import RetrievalQuery
from core.types import Provenance, RetrievedChunk, RetrieverKind


class PgVectorRetriever:
    """Implements `Retriever` against the `chunks` table."""

    name = "pgvector_semantic"

    def __init__(self, session: AsyncSession, embeddings: EmbeddingPipeline | None = None) -> None:
        self._session = session
        self._embeddings = embeddings

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Embed the query, then search (spec section 70's `Retriever`).

        Requires an embedding pipeline. `retrieve_with_vector` stays available
        for callers that already hold the vector and should not pay to compute
        it twice.
        """
        if self._embeddings is None:
            raise ProviderError(
                "Semantic retrieval needs an embedding pipeline.",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
                provider="mistral",
            )
        vector = await self._embeddings.embed_query(query.text)
        return await self.retrieve_with_vector(query, vector)

    async def retrieve_with_vector(
        self, query: RetrievalQuery, vector: list[float]
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks to an already-embedded query.

        Takes the vector rather than the text so the caller controls when the
        embedding request happens — it is the slow, billable part, and the
        trace records it as its own stage.
        """
        distance = Chunk.embedding.cosine_distance(vector)
        statement = (
            select(Chunk, distance.label("distance"))
            .where(
                Chunk.project_id == query.project_id,
                # Chunks awaiting embedding are simply not dense-retrievable;
                # the index would reject them anyway.
                Chunk.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(query.top_k)
        )

        rows = (await self._session.execute(statement)).all()
        return [
            RetrievedChunk(
                text=chunk.text,
                # Cosine distance is in [0, 2]; similarity is the complement,
                # so a larger score is a better match for every retriever.
                score=1.0 - float(distance),
                retriever=RetrieverKind.SEMANTIC,
                provenance=Provenance(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                ),
                rank=rank,
                metadata={"embedding_model": chunk.embedding_model},
            )
            for rank, (chunk, distance) in enumerate(rows)
        ]
