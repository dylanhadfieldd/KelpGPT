# backend/db.py
"""
Database bootstrap for KelpGPT (Windows/SQLite-ready, Postgres-capable).

- Defaults to SQLite at db/kelpgpt.sqlite (gitignored)
- Safe to use with Streamlit (check_same_thread=False)
- Provides:
    - engine (SQLAlchemy Engine)
    - SessionLocal (sessionmaker)
    - init_db() to create tables
    - get_db() context manager for 'with' usage
"""

from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

# Local utilities & models
# Ensure utils.paths exists per the repo design
from utils.paths import DB_DIR
from .models import Base


def _resolve_database_url() -> str:
    """
    Prefer DATABASE_URL; otherwise use local SQLite file under db/kelpgpt.sqlite.
    Examples:
      - sqlite:///db/kelpgpt.sqlite
      - postgresql+psycopg://user:pass@host:5432/kelpgpt
    """
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    # Default to SQLite in repo's db/ folder
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(DB_DIR / 'kelpgpt.sqlite').as_posix()}"


def _engine_kwargs_for_url(url: string) -> dict:
    """
    Tweak engine kwargs by backend. For SQLite + Streamlit, we need check_same_thread=False.
    """
    if url.startswith("sqlite:///"):
        return {"connect_args": {"check_same_thread": False}}
    # Postgres/MySQL: no special kwargs by default
    return {}


# Build engine + sessionmaker
DATABASE_URL = _resolve_database_url()
_engine_kwargs = _engine_kwargs_for_url(DATABASE_URL)
engine: Engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """
    Create all tables if they don't exist.
    Call this once on app startup (e.g., top of app/app.py).
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


@contextmanager
def get_db() -> Iterator[Session]:
    """
    Context manager for a short-lived DB session:
        with get_db() as db:
            db.query(...)

    Ensures close() is called even on exceptions.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

