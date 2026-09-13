"""
Database initialization — creates all tables.
"""

import logging

from app.db.database import async_engine, Base
from app.db import models  # noqa: F401 — ensures models are registered

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Create all database tables if they don't exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified successfully.")


async def drop_db() -> None:
    """Drop all database tables (use only in development)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All database tables dropped.")
