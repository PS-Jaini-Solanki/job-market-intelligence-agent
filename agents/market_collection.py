"""
agents/market_collection.py
============================
Market Collection Agent — fetches real job postings from APIs.

RESPONSIBILITY:
- Call the job_search_tool to fetch postings from Adzuna + Arbeitnow
- Store results in the graph state for downstream agents

This agent does NOT analyze the data — it only collects it.
Analysis is the job of SkillExtractionAgent and TrendAnalysisAgent.

SEPARATION OF CONCERNS:
Collection → Extraction → Analysis → Report
Each step is a separate agent, making the system easier to test,
debug, and modify independently.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState
from tools.job_search_tool import _search_adzuna, _search_arbeitnow
from models.job import JobPosting, JobSearchResult

logger = logging.getLogger(__name__)


async def run_market_collection(state: GraphState) -> dict:
    """
    Market Collection agent node function.

    Fetches job postings from configured APIs and stores them in state.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with job_search_results populated.
    """
    plan = state.get("execution_plan", {})
    role = state.get("role", "AI Engineer")
    country = state.get("country_filter")
    settings_max = 100  # default

    # Import here to avoid circular imports
    from config import get_settings
    settings_max = get_settings().MAX_JOB_POSTINGS

    logger.info(
        "Market collection starting",
        extra={"role": role, "country": country, "max": settings_max},
    )

    per_source = settings_max // 2

    # Collect from Adzuna
    adzuna_jobs: list[JobPosting] = []
    if country:
        adzuna_jobs = await _search_adzuna(
            role=role,
            country_code=country,
            max_results=per_source,
        )
    else:
        # Try a couple of major markets if no country specified
        for c in ["US", "IN", "GB"]:
            jobs = await _search_adzuna(
                role=role,
                country_code=c,
                max_results=per_source // 3,
            )
            adzuna_jobs.extend(jobs)

    # Collect from Arbeitnow
    arbeitnow_jobs = await _search_arbeitnow(
        role=role,
        max_results=per_source,
    )

    all_postings = adzuna_jobs + arbeitnow_jobs
    sources = []
    if adzuna_jobs:
        sources.append("adzuna")
    if arbeitnow_jobs:
        sources.append("arbeitnow")

    result = JobSearchResult(
        query=role,
        postings=all_postings,
        total_found=len(all_postings),
        sources_queried=sources,
        search_params={"role": role, "country": country or "global"},
    )

    logger.info(
        "Market collection complete",
        extra={"total_postings": len(all_postings), "sources": sources},
    )

    return {
        "job_search_results": result.model_dump(),
        "completed_agents": state.get("completed_agents", []) + ["market_collection"],
        "messages": [
            AIMessage(
                content=(
                    f"🔍 Collected **{len(all_postings)}** job postings "
                    f"for '{role}' from {', '.join(sources) or 'no sources configured'}."
                )
            )
        ],
    }
