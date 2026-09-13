"""
AI MADAC — FastAPI Application Entry Point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.init_db import init_db
from app.utils.logging import setup_logging

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("Starting AI MADAC Backend...")

    # Ensure required directories exist
    settings.ensure_directories()

    # Initialize database tables
    await init_db()

    logger.info("AI MADAC Backend ready.")
    yield
    logger.info("Shutting down AI MADAC Backend...")


# Create FastAPI application
app = FastAPI(
    title="AI MADAC",
    description="Autonomous Multi-Agent Data Science Analyst",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for generated charts
app.mount("/static/charts", StaticFiles(directory=settings.CHARTS_DIR, check_dir=False), name="charts")

# --- Register API Routers ---
from app.api.auth import router as auth_router  # noqa: E402
from app.api.datasets import router as datasets_router  # noqa: E402
from app.api.queries import router as queries_router  # noqa: E402
from app.api.ml import router as ml_router  # noqa: E402
from app.api.reports import router as reports_router  # noqa: E402

app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(queries_router)
app.include_router(ml_router)
app.include_router(reports_router)





@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI MADAC Backend"}
