"""Alembic environment.

The database URL comes from application settings, and the target metadata from
`app.models`, so migrations and the ORM can never drift apart.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any, Literal

from alembic.autogenerate.api import AutogenContext
from pgvector.sqlalchemy import Vector
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import get_settings
from app.models import Base  # imports every model module

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Respect a URL injected by the caller (the test harness does this); fall
# back to application settings so the CLI needs no arguments.
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", str(get_settings().DATABASE_URL))
target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: AutogenContext) -> str | Literal[False]:
    """Render pgvector columns together with their import.

    Autogenerate otherwise emits a bare `pgvector.sqlalchemy...VECTOR(...)`
    reference and no import, producing a migration that raises NameError.
    Returning False defers every other type to the default renderer.
    """
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({obj.dim})"
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_item=render_item,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=None,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
