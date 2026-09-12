"""Typed application settings.

Every value is read from the environment (see `.env.example` at the repository
root). Nothing in the codebase should call `os.environ` directly — import
`get_settings()` instead so configuration stays typed and testable.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class AuthMode(StrEnum):
    """Phase 1 runs with authentication disabled (spec section 43: do not
    over-engineer multi-tenancy initially). `JWT` is wired in a later phase."""

    DISABLED = "disabled"
    JWT = "jwt"


class VectorStoreBackend(StrEnum):
    PGVECTOR = "pgvector"
    QDRANT = "qdrant"


class KeywordBackend(StrEnum):
    POSTGRES_FTS = "postgres_fts"
    BM25 = "bm25"


class ObjectStorageBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"


class Settings(BaseSettings):
    """Application configuration, validated at startup."""

    model_config = SettingsConfigDict(
        # Resolution order: backend/.env then repo-root .env (root wins).
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # --- application ---
    PROJECT_NAME: str = "Adaptive Evidence AI Lab"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    SECRET_KEY: SecretStr = SecretStr("change-me")

    # --- logging / observability (spec section 48) ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: LogFormat = LogFormat.CONSOLE

    # --- auth ---
    AUTH_MODE: AuthMode = AuthMode.DISABLED

    # --- api ---
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = Field(default_factory=list)

    # --- database ---
    DATABASE_URL: PostgresDsn
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 5
    DATABASE_ECHO: bool = False

    # --- cache / queue ---
    REDIS_URL: RedisDsn

    # --- model providers ---
    # Spec section 81: only include keys for providers actually implemented.
    # An empty key disables the provider rather than failing at import.
    MISTRAL_API_KEY: SecretStr | None = None
    MISTRAL_API_BASE: str = "https://api.mistral.ai/v1"
    MODEL_REQUEST_TIMEOUT_SECONDS: float = 60.0
    MODEL_MAX_RETRIES: int = 3

    # --- retrieval ---
    VECTOR_STORE_BACKEND: VectorStoreBackend = VectorStoreBackend.PGVECTOR
    KEYWORD_BACKEND: KeywordBackend = KeywordBackend.POSTGRES_FTS
    EMBEDDING_MODEL: str = "mistral-embed"
    # mistral-embed emits fixed 1024-dimensional vectors; the value is not
    # reducible. It must equal the pgvector column width, which is enforced by
    # `validate_runtime_settings` — changing models requires a migration.
    EMBEDDING_DIMENSIONS: int = 1024
    # Mistral accepts up to 512 inputs per embeddings call.
    EMBEDDING_BATCH_SIZE: int = 128
    RETRIEVAL_DEFAULT_TOP_K: int = 10

    # --- ingestion (spec section 10) ---
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
    ALLOWED_UPLOAD_MIME_TYPES: set[str] = Field(
        default_factory=lambda: {
            "application/pdf",
            "text/plain",
            "text/markdown",
        }
    )
    CHUNK_SIZE_CHARS: int = 1200
    CHUNK_OVERLAP_CHARS: int = 150

    # --- object storage ---
    OBJECT_STORAGE_BACKEND: ObjectStorageBackend = ObjectStorageBackend.LOCAL
    OBJECT_STORAGE_ENDPOINT: str | None = None
    OBJECT_STORAGE_BUCKET: str | None = None
    OBJECT_STORAGE_ACCESS_KEY: SecretStr | None = None
    OBJECT_STORAGE_SECRET_KEY: SecretStr | None = None
    OBJECT_STORAGE_LOCAL_PATH: Path = Path("./.data/objects")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT is Environment.PRODUCTION

    @field_validator(
        "MISTRAL_API_KEY",
        "OBJECT_STORAGE_ACCESS_KEY",
        "OBJECT_STORAGE_SECRET_KEY",
        mode="before",
    )
    @classmethod
    def _blank_secret_means_unset(cls, value: object) -> object:
        """Treat an empty variable as absent.

        `.env` files carry placeholder lines like `MISTRAL_API_KEY=`, which
        would otherwise parse as an empty secret — indistinguishable from a
        configured key until the provider rejects it with a 401.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("SECRET_KEY")
    @classmethod
    def _reject_placeholder_secret(cls, value: SecretStr) -> SecretStr:
        """Fail fast rather than run production on the template secret."""
        if value.get_secret_value().startswith("change-me"):
            # Tolerated in development; enforced for real deployments by
            # `validate_runtime_settings()` below, which sees ENVIRONMENT too.
            return value
        return value


def validate_runtime_settings(settings: Settings) -> None:
    """Refuse to boot with unsafe or inconsistent configuration."""
    # Checked in every environment: a mismatch here writes vectors the index
    # cannot hold, and the failure would otherwise surface deep in a worker.
    from app.models.chunk import EMBEDDING_DIMENSIONS as COLUMN_DIMENSIONS

    if settings.EMBEDDING_DIMENSIONS != COLUMN_DIMENSIONS:
        raise RuntimeError(
            f"EMBEDDING_DIMENSIONS ({settings.EMBEDDING_DIMENSIONS}) does not match the "
            f"chunks.embedding column width ({COLUMN_DIMENSIONS}). Changing embedding "
            "model requires a migration that alters the column and rebuilds the index."
        )

    if not settings.is_production:
        return

    problems: list[str] = []
    if settings.SECRET_KEY.get_secret_value().startswith("change-me"):
        problems.append("SECRET_KEY is still the template value")
    if settings.DEBUG:
        problems.append("DEBUG must be false in production")
    if settings.LOG_FORMAT is not LogFormat.JSON:
        problems.append("LOG_FORMAT must be 'json' in production")
    if settings.AUTH_MODE is AuthMode.DISABLED:
        problems.append("AUTH_MODE must not be 'disabled' in production")
    if settings.MISTRAL_API_KEY is None:
        problems.append("MISTRAL_API_KEY must be set in production")
    if problems:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Override via dependency injection in tests."""
    return Settings()
