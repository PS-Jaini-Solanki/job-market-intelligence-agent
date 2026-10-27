"""
api/routes/health.py
=====================
Health check endpoints.

These endpoints are used by:
- Docker health checks
- Monitoring systems (Prometheus, Datadog, etc.)
- Load balancers to determine if the app is ready

A well-designed health check tells you:
1. Is the API alive? (basic /health)
2. Is the database connected? (deep /health/ready)
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Track when the app started (for uptime reporting)
_startup_time = datetime.utcnow()


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    """Response model for the readiness check endpoint."""
    status: str
    database: str
    llm_configured: bool
    llm_model: str
    apis_configured: dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check — is the API alive?

    Returns 200 OK if the application is running.
    Used by Docker health checks and load balancers.
    """
    uptime = (datetime.utcnow() - _startup_time).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=round(uptime, 2),
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """
    Deep readiness check — are all dependencies available?

    Checks:
    - Database connection
    - LLM API key configured
    - External API keys configured

    Returns 200 if ready, 503 if not.
    """
    from config import get_settings
    settings = get_settings()

    # Check database
    db_status = "unknown"
    try:
        from database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
        logger.warning("Database health check failed", extra={"error": str(e)})

    # Check the configured model actually exists for this key: a key alone
    # says nothing once a provider retires a model (every LLM call then
    # fails and the agents silently run on their fallbacks).
    llm_status = "not configured"
    if settings.GROQ_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"https://api.groq.com/openai/v1/models/{settings.GROQ_MODEL}",
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                )
            llm_status = (
                f"{settings.GROQ_MODEL}: available" if response.status_code == 200
                else f"{settings.GROQ_MODEL}: unavailable (HTTP {response.status_code})"
            )
        except Exception as e:
            llm_status = f"{settings.GROQ_MODEL}: error: {str(e)[:50]}"
            logger.warning("LLM health check failed", extra={"error": str(e)})

    healthy = db_status == "connected" and llm_status.endswith("available")
    return ReadyResponse(
        status="ready" if healthy else "degraded",
        database=db_status,
        llm_configured=bool(settings.GROQ_API_KEY),
        llm_model=llm_status,
        apis_configured={
            "github": settings.has_github,
            "tavily": settings.has_tavily,
            "adzuna": settings.has_adzuna,
            "reddit": settings.has_reddit,
        },
    )
