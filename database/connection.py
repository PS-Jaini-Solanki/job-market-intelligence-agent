"""
database/connection.py
======================
Async PostgreSQL connection setup using SQLAlchemy 2.0.

Key design decisions:
1. We use async SQLAlchemy so the FastAPI endpoints don't block.
2. The engine is created once at startup (singleton pattern).
3. Each request gets its own session via the get_session() dependency.

How to use in FastAPI:
    from database.connection import get_session

    @app.get("/example")
    async def example(session: AsyncSession = Depends(get_session)):
        result = await session.execute(select(User))
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import get_settings

logger = logging.getLogger(__name__)

# ============================================================
# SQLAlchemy Base
# ============================================================
# All ORM models must inherit from this Base.
# It provides metadata tracking and relationships.


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ============================================================
# Engine & Session Factory (module-level singletons)
# ============================================================
# These are created lazily (on first call to get_engine())
# so we don't need a database connection at import time.

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Return the async SQLAlchemy engine (creates it on first call).

    The engine manages the connection pool and should be shared
    across the entire application.

    Returns:
        The singleton AsyncEngine instance.
    """
    global _engine

    if _engine is None:
        settings = get_settings()
        logger.info(
            "Creating SQLAlchemy async engine",
            extra={"database_url": settings.database_url_masked},
        )
        _engine = create_async_engine(
            settings.SQLALCHEMY_DATABASE_URL,
            # Pool settings
            pool_size=10,          # Number of connections to keep open
            max_overflow=20,       # Extra connections allowed under load
            pool_pre_ping=True,    # Test connections before using them
            echo=settings.API_DEBUG,  # Log all SQL queries in debug mode
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Return the session factory (creates it on first call).

    The session factory creates new sessions from the shared engine.

    Returns:
        The singleton async_sessionmaker instance.
    """
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,  # Keep objects usable after commit
            autoflush=False,          # Don't auto-flush (be explicit)
        )

    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Use this for non-FastAPI code (agents, tools, background tasks).

    Usage:
        async with get_db_session() as session:
            result = await session.execute(select(User))

    The session is automatically committed on success and rolled back
    on any exception.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in a FastAPI route:
        @router.get("/example")
        async def example(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with get_db_session() as session:
        yield session


async def create_all_tables() -> None:
    """
    Create all database tables defined in the ORM models.

    This is called during application startup. In production, Alembic
    migrations handle schema changes, but this is useful for testing.
    """
    engine = get_engine()
    # Import all models so SQLAlchemy knows about them
    import database.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created successfully")


async def close_engine() -> None:
    """
    Close the database engine (called during application shutdown).

    This cleanly closes all pooled connections.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine closed")
