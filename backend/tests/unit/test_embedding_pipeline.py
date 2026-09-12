"""Batching pipeline invariants."""

from __future__ import annotations

import pytest

from core.embeddings.pipeline import BatchedEmbeddingPipeline
from tests.fakes import FakeEmbeddingModel, deterministic_vector

pytestmark = pytest.mark.unit


def _pipeline(
    batch_size: int = 3, max_batch_size: int = 8
) -> tuple[BatchedEmbeddingPipeline, FakeEmbeddingModel]:
    model = FakeEmbeddingModel(max_batch_size=max_batch_size)
    return BatchedEmbeddingPipeline(model, batch_size=batch_size), model


async def test_texts_are_split_into_batches() -> None:
    pipeline, model = _pipeline(batch_size=3)

    vectors = await pipeline.embed_texts([f"text {i}" for i in range(7)])

    assert len(vectors) == 7
    assert [len(call) for call in model.calls] == [3, 3, 1]


async def test_order_is_preserved_across_batches() -> None:
    """Vectors are attached to chunks positionally, so a reordering here would
    mis-attribute every embedding."""
    pipeline, _ = _pipeline(batch_size=2)
    texts = [f"unique text {i}" for i in range(5)]

    vectors = await pipeline.embed_texts(texts)

    assert vectors == [deterministic_vector(text) for text in texts]


async def test_batch_size_never_exceeds_the_provider_limit() -> None:
    """Configuration must not be able to push a batch past what the provider
    accepts; the fake asserts on the limit."""
    pipeline, model = _pipeline(batch_size=1000, max_batch_size=4)

    await pipeline.embed_texts([f"text {i}" for i in range(9)])

    assert max(len(call) for call in model.calls) == 4


async def test_empty_input_makes_no_call() -> None:
    pipeline, model = _pipeline()

    assert await pipeline.embed_texts([]) == []
    assert model.calls == []


async def test_query_uses_the_same_model_as_the_corpus() -> None:
    """A query embedded by another model would not share a vector space with
    the chunks, making similarity scores meaningless."""
    pipeline, model = _pipeline()

    vector = await pipeline.embed_query("a question")

    assert vector == deterministic_vector("a question")
    assert pipeline.model_id == model.model_id


async def test_dimensions_come_from_the_model() -> None:
    pipeline, model = _pipeline()

    assert pipeline.dimensions == model.dimensions


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ValueError):
        BatchedEmbeddingPipeline(FakeEmbeddingModel(), batch_size=batch_size)


async def test_closing_the_pipeline_closes_the_model() -> None:
    pipeline, model = _pipeline()

    await pipeline.aclose()

    assert model.closed is True
