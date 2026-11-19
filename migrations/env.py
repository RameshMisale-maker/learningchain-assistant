"""Alembic environment: migrates the app's database (settings.database_url) to the ORM metadata."""
from __future__ import annotations

import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db.models import Base  # noqa: E402
from config.settings import get_settings  # noqa: E402

config = context.config
target_metadata = Base.metadata


def _configure(**kwargs) -> None:
    # render_as_batch: SQLite cannot ALTER most columns in place - batch mode recreates the table safely
    context.configure(target_metadata=target_metadata, render_as_batch=True, compare_type=True, **kwargs)


def run_migrations_offline() -> None:
    _configure(url=config.get_main_option("sqlalchemy.url") or get_settings().database_url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")       # passed in by backend/db/session.py at startup
    if connection is not None:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(config.get_main_option("sqlalchemy.url") or get_settings().database_url, poolclass=pool.NullPool)
    with engine.connect() as conn:
        _configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
