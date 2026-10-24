"""
Database initialization — creates all tables.
"""

import logging

from app.db.database import async_engine, Base
from app.db import models  # noqa: F401 — ensures models are registered

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Create all database tables if they don't exist and run non-destructive schema migrations."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Non-destructive migration for existing PostgreSQL databases:
        # PostgreSQL enforces character length strictly; ensure agent_runs.status is VARCHAR(50).
        # SQLite handles VARCHAR dynamically and does not require ALTER TABLE.
        drivername = getattr(async_engine.url, "drivername", "")
        if "postgresql" in drivername:
            try:
                from sqlalchemy import text
                await conn.execute(
                    text("ALTER TABLE agent_runs ALTER COLUMN status TYPE VARCHAR(50);")
                )
                logger.info("Database migration: agent_runs.status column widened to VARCHAR(50).")
            except Exception as e:
                logger.warning("agent_runs.status migration note: %s", e)

    logger.info("Database tables created/verified successfully.")


async def drop_db() -> None:
    """Drop all database tables (use only in development)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All database tables dropped.")
