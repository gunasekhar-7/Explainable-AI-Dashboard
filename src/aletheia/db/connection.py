"""Database connection, session management, and initialization.

Provides connection pooling, engine lifecycle, and FastAPI dependency injection.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from aletheia.core.config import get_settings
from aletheia.db.models import Base


def create_app_engine(url: str, *, echo: bool = False) -> Engine:
    """Create a configured SQLAlchemy engine."""
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    elif url.startswith("postgresql"):
        connect_args["connect_timeout"] = 5

    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Retrieve or create an engine for the given URL or current settings."""
    resolved_url = url
    if resolved_url is None:
        settings = get_settings()
        resolved_url = settings.database_url
    if not resolved_url:
        resolved_url = "sqlite:///aletheia.db"
    return create_app_engine(resolved_url)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Build a sessionmaker bound to the given engine."""
    target_engine = engine or get_engine()
    return sessionmaker(bind=target_engine, autocommit=False, autoflush=False)


def init_db(engine: Engine | None = None) -> None:
    """Create all registered database tables."""
    target_engine = engine or get_engine()
    Base.metadata.create_all(bind=target_engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a transactional database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
