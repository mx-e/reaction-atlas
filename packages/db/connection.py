"""Database connection management."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_session_factory = None


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


def get_engine(database_url: str | None = None, pool_size: int = 5):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = database_url or get_database_url()
        _engine = create_engine(
            url,
            pool_size=pool_size,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(database_url)
        _session_factory = sessionmaker(bind=engine)
    return _session_factory


def get_session(database_url: str | None = None) -> Session:
    """Create a new database session."""
    factory = get_session_factory(database_url)
    return factory()


def init_db(database_url: str | None = None):
    """Create all tables in the database."""
    from packages.db.models import Base

    engine = get_engine(database_url)
    Base.metadata.create_all(engine)


def reset_connection():
    """Reset the global engine and session factory (for testing)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
