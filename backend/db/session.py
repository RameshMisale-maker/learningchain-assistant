"""Database engine / session factory. Works with PostgreSQL (DATABASE_URL) or SQLite (default)."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base
from config.settings import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_ENGINE_LOCK = threading.Lock()   # the approved-answer warm-up thread and the first request may start together


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    with _ENGINE_LOCK:
        if _engine is not None:          # another thread finished the set-up while this one waited
            return _engine
        url = get_settings().database_url
        kwargs: dict = {"pool_pre_ping": True, "future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - driver hook
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        migrate(engine)
        # published only when fully migrated, so no thread ever uses a half-set-up engine
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        _engine = engine
    return _engine


def migrate(engine: Engine) -> None:
    """Bring the schema to the latest Alembic migration (migrations/versions). A database created before
    migrations existed (tables but no alembic_version) is first stamped at the baseline 0001 - its data is untouched."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from config.settings import BASE_DIR

    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
    sqlite = engine.url.get_backend_name() == "sqlite"
    with engine.connect() as conn:
        if sqlite:   # table rebuilds (batch mode) must not cascade-delete child rows
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            conn.commit()
        with conn.begin():
            cfg.attributes["connection"] = conn
            tables = set(inspect(conn).get_table_names())
            if "alembic_version" not in tables and "documents" in tables:
                command.stamp(cfg, "0001")
            command.upgrade(cfg, "head")
        if sqlite:
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
            conn.commit()


@contextmanager
def db_session() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose the engine (tests / settings changes)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
