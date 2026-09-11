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

    # --- model providers (empty means "provider disabled") ---
    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None

    # --- retrieval ---
    VECTOR_STORE_BACKEND: VectorStoreBackend = VectorStoreBackend.PGVECTOR
    KEYWORD_BACKEND: KeywordBackend = KeywordBackend.POSTGRES_FTS
    EMBEDDING_MODEL: str | None = None
    EMBEDDING_DIMENSIONS: int = 1536

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
    """Refuse to boot a production deployment with unsafe configuration."""
    if not settings.is_production:
        return

    problems: list[str] = []
    if settings.SECRET_KEY.get_secret_value().startswith("change-me"):
        problems.append("SECRET_KEY is still the template value")
    if settings.DEBUG:
        problems.append("DEBUG must be false in production")
    if settings.LOG_FORMAT is not LogFormat.JSON:
        problems.append("LOG_FORMAT must be 'json' in production")
    if problems:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Override via dependency injection in tests."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
