"""
api/routes/query.py
====================
Market intelligence query API endpoints.

POST /api/query  — Submit a market intelligence query
GET  /api/query/{id} — Get the result of a previous query

The query endpoint:
1. Validates the request
2. Creates a database record
3. Runs the LangGraph agent graph
4. Saves and returns the result
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.graph import get_graph
from agents.state import create_initial_state
from database.connection import get_session
from database.repositories import QueryRepository

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================
# Request / Response Models
# ============================================================


class QueryRequest(BaseModel):
    """Request body for submitting a market intelligence query."""

    query: str = Field(
        description="Natural language query about the job market",
        examples=[
            "What skills are required for AI Engineers in India?",
            "Should I learn MCP or LangGraph?",
            "Compare AI job markets in India vs USA",
        ],
        min_length=5,
        max_length=500,
    )
    country: Optional[str] = Field(
        default=None,
        description="ISO country code to focus on (e.g., 'IN', 'US')",
        examples=["IN", "US", "GB"],
    )
    role: Optional[str] = Field(
        default=None,
        description="Job role to analyze (e.g., 'AI Engineer')",
    )
    technologies: Optional[list[str]] = Field(
        default=None,
        description="Specific technologies to analyze trends for",
    )


class QueryResponse(BaseModel):
    """Response from a query submission."""

    query_id: str
    status: str
    message: str


class QueryResult(BaseModel):
    """Full result of a completed query."""

    query_id: str
    query: str
    status: str
    result: Optional[dict] = None
    execution_time_seconds: Optional[float] = None
    error: Optional[str] = None


# ============================================================
# Endpoints
# ============================================================


@router.post(
    "/query",
    response_model=QueryResult,
    status_code=status.HTTP_200_OK,
    summary="Submit a market intelligence query",
    description="Runs the full agent pipeline and returns the market intelligence report.",
)
async def submit_query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> QueryResult:
    """
    Submit a market intelligence query and get results synchronously.

    The agent pipeline typically takes 20–60 seconds to complete.
    For a non-blocking version, use the Chainlit UI which shows
    progress in real-time.

    Args:
        request: Query parameters.
        session: Database session (injected by FastAPI).

    Returns:
        QueryResult with the complete report.
    """
    query_repo = QueryRepository(session)

    # Create database record
    query_record = await query_repo.create(
        query=request.query,
        query_type="market_intelligence",
    )
    query_id = query_record.id

    await query_repo.mark_running(query_id)

    start_time = time.monotonic()

    try:
        # Build initial state for the graph
        initial_state = create_initial_state(
            user_query=request.query,
            country_filter=request.country,
            role=request.role,
            technologies=request.technologies,
            query_id=query_id,
        )

        # Run the LangGraph agent pipeline
        graph = get_graph()
        logger.info("Starting agent graph execution", extra={"query_id": query_id})
        final_state = await graph.ainvoke(initial_state)

        execution_time = round(time.monotonic() - start_time, 2)

        # Extract the report from state
        report = final_state.get("final_report")
        if not report:
            raise ValueError("Agent pipeline did not produce a report")

        # Include the concise, question-aligned answer alongside the report
        # so API consumers get a direct response too.
        result = dict(report)
        if final_state.get("direct_answer"):
            result["direct_answer"] = final_state["direct_answer"]

        # Save to database
        await query_repo.mark_completed(
            query_id=query_id,
            result=report,
            execution_time=execution_time,
        )

        logger.info(
            "Query completed",
            extra={"query_id": query_id, "execution_time": execution_time},
        )

        return QueryResult(
            query_id=query_id,
            query=request.query,
            status="completed",
            result=report,
            execution_time_seconds=execution_time,
        )

    except Exception as e:
        execution_time = round(time.monotonic() - start_time, 2)
        error_msg = str(e)
        logger.error(
            "Query failed",
            extra={"query_id": query_id, "error": error_msg},
        )
        await query_repo.mark_failed(query_id=query_id, error=error_msg)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {error_msg}",
        )


@router.get(
    "/query/{query_id}",
    response_model=QueryResult,
    summary="Get query result by ID",
)
async def get_query_result(
    query_id: str,
    session: AsyncSession = Depends(get_session),
) -> QueryResult:
    """
    Retrieve the result of a previously submitted query.

    Args:
        query_id: The query ID returned by POST /api/query.
        session: Database session.

    Returns:
        QueryResult with status and report (if completed).
    """
    query_repo = QueryRepository(session)
    record = await query_repo.get_by_id(query_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' not found",
        )

    return QueryResult(
        query_id=record.id,
        query=record.query,
        status=record.status,
        result=record.result,
        execution_time_seconds=record.execution_time,
        error=record.result.get("error") if record.result else None,
    )
