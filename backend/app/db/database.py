"""
Database connection and session management.
Supports both SQLite and PostgreSQL backends seamlessly.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

is_sqlite = settings.DATABASE_URL.startswith("sqlite")

async_kwargs = {}
sync_kwargs = {}

if is_sqlite:
    sync_kwargs["connect_args"] = {"check_same_thread": False}
else:
    async_kwargs.update({"pool_size": 10, "max_overflow": 20, "pool_pre_ping": True})
    sync_kwargs.update({"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True})

# Async engine for FastAPI endpoints
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    **async_kwargs,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for dataset operations (Pandas to_sql)
sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    echo=False,
    **sync_kwargs,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """Dependency that provides an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
