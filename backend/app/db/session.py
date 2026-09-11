"""Async engine and session factory lifecycle."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    """Create the engine and session factory. Called once from the app lifespan."""
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_async_engine(
            str(settings.DATABASE_URL),
            echo=settings.DATABASE_ECHO,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_pre_ping=True,
        )
        _sessionmaker = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("Database engine is not initialised; call init_engine() first.")
    return _sessionmaker


async def dispose_engine() -> None:
    """Close pooled connections on shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
