"""Model provider selection — the composition root for models.

Lives in `app` because it reads `Settings`. Callers depend only on the
`EmbeddingPipeline` protocol, so adding a provider touches this module alone
(spec rule 6: keep provider-specific logic isolated).
"""

from __future__ import annotations

from app.core.config import Settings
from core.embeddings.pipeline import BatchedEmbeddingPipeline
from core.errors import ErrorCode, ProviderError
from core.reasoning.grounded import GroundedAnswerGenerator
from models.providers.mistral import (
    MISTRAL_EMBED_MODEL,
    MistralChatModel,
    MistralEmbeddingModel,
)


def generation_enabled(settings: Settings) -> bool:
    """Whether a chat provider is configured for answering."""
    return settings.MISTRAL_API_KEY is not None


def get_answer_generator(settings: Settings) -> GroundedAnswerGenerator:
    """Build the configured answer generator.

    The caller owns the result and must `aclose()` it — it holds an HTTP
    connection pool.
    """
    if settings.MISTRAL_API_KEY is None:
        raise ProviderError(
            "MISTRAL_API_KEY is not set, so answers cannot be generated.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
        )
    model = MistralChatModel(
        api_key=settings.MISTRAL_API_KEY.get_secret_value(),
        base_url=settings.MISTRAL_API_BASE,
        model_id=settings.GENERATION_MODEL,
        timeout=settings.MODEL_REQUEST_TIMEOUT_SECONDS,
        max_retries=settings.MODEL_MAX_RETRIES,
    )
    return GroundedAnswerGenerator(model, temperature=settings.GENERATION_TEMPERATURE)


def embeddings_enabled(settings: Settings) -> bool:
    """Whether an embedding provider is configured.

    Without a key the platform still ingests, parses and chunks; only the
    embedding stage is unavailable. That keeps the pipeline usable offline
    rather than failing the whole upload.
    """
    return settings.MISTRAL_API_KEY is not None


def get_embedding_pipeline(settings: Settings) -> BatchedEmbeddingPipeline:
    """Build the configured embedding pipeline.

    The caller owns the returned pipeline and must `aclose()` it — it holds an
    HTTP connection pool.
    """
    if settings.MISTRAL_API_KEY is None:
        raise ProviderError(
            "MISTRAL_API_KEY is not set, so documents cannot be embedded.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
        )

    if settings.EMBEDDING_MODEL != MISTRAL_EMBED_MODEL:
        raise ProviderError(
            f"Unknown embedding model {settings.EMBEDDING_MODEL!r}.",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            provider="mistral",
            details={"supported": [MISTRAL_EMBED_MODEL]},
        )

    model = MistralEmbeddingModel(
        api_key=settings.MISTRAL_API_KEY.get_secret_value(),
        base_url=settings.MISTRAL_API_BASE,
        model_id=settings.EMBEDDING_MODEL,
        timeout=settings.MODEL_REQUEST_TIMEOUT_SECONDS,
        max_retries=settings.MODEL_MAX_RETRIES,
    )
    return BatchedEmbeddingPipeline(model, batch_size=settings.EMBEDDING_BATCH_SIZE)
