"""
api/main.py
============
FastAPI application factory and startup configuration.

WHAT THIS FILE DOES:
1. Creates the FastAPI app instance
2. Configures CORS (Cross-Origin Resource Sharing)
3. Sets up lifespan events (startup / shutdown)
4. Registers all API routers
5. Adds global exception handling

DESIGN NOTE:
We use the "application factory" pattern — the app is created by
calling create_app() rather than at module level. This makes it
easier to create different app instances for testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable no matter how the API is launched
# (uvicorn, `python api/main.py`, Docker, etc.).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings

# Import routers
from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.routes.resume import router as resume_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.

    This is FastAPI's modern way to handle startup/shutdown events.
    """
    # === STARTUP ===
    settings = get_settings()
    logger.info(f"Starting Job Market Intelligence Agent API on port {settings.API_PORT}")

    # Create database tables if they don't exist
    try:
        from database.connection import create_all_tables
        await create_all_tables()
        logger.info("Database tables ready")
    except Exception as e:
        logger.error(f"Database setup failed: {e}. Continuing without database.")

    # Create upload directories
    import os
    os.makedirs(settings.RESUME_UPLOAD_DIR, exist_ok=True)
    os.makedirs("./reports", exist_ok=True)

    logger.info("API startup complete")

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info("Shutting down...")
    try:
        from database.connection import close_engine
        await close_engine()
    except Exception:
        pass
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Job Market Intelligence Agent",
        description=(
            "AI-powered job market analysis using Agentic AI, MCP, and LangGraph. "
            "Analyzes real job postings to provide evidence-based career insights."
        ),
        version="1.0.0",
        docs_url="/docs",          # Swagger UI
        redoc_url="/redoc",        # ReDoc UI
        lifespan=lifespan,
    )

    # --- CORS ---
    # Allow the Chainlit frontend to call this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8001", "http://127.0.0.1:8001", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register Routers ---
    app.include_router(health_router, tags=["Health"])
    app.include_router(query_router, prefix="/api", tags=["Market Intelligence"])
    app.include_router(resume_router, prefix="/api", tags=["Resume"])

    # --- Global Exception Handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch all unhandled exceptions and return a clean JSON error.

        In production, this prevents stack traces from leaking to clients.
        """
        logger.error(
            "Unhandled exception",
            extra={"path": request.url.path, "error": str(exc)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.API_DEBUG else "See server logs for details",
            },
        )

    return app


# Create the app instance (used by uvicorn)
app = create_app()
