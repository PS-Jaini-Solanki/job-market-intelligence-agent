"""
agents/market_collection.py
============================
Market Collection Agent — fetches real job postings from APIs.

RESPONSIBILITY:
- Call the `jobs` MCP server (search_jobs) to fetch postings from
  Adzuna + Arbeitnow via the MCP registry
- Record the MCP call so the UI can show the exact routing
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
from typing import Optional

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.agent import MCPCallRecord
from models.job import JobSearchResult

logger = logging.getLogger(__name__)


async def run_market_collection(state: GraphState) -> dict:
    """
    Market Collection agent node function.

    Fetches job postings through the jobs MCP server and stores them in state.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with job_search_results populated.
    """
    role = state.get("role") or "Software Engineer"
    country = state.get("country_filter")
    countries = state.get("countries") or []

    from config import get_settings
    settings_max = get_settings().MAX_JOB_POSTINGS

    # For multi-country queries (e.g. "Compare India vs USA"), collect postings
    # for EVERY country the user mentioned so the comparison has real data for
    # each market. Otherwise fall back to the single country filter.
    countries_to_collect = [c.upper() for c in countries if isinstance(c, str) and c]
    if not countries_to_collect and country:
        countries_to_collect = [country.upper()]
    if not countries_to_collect:
        countries_to_collect = ["US"]

    logger.info(
        "Market collection starting",
        extra={"role": role, "countries": countries_to_collect, "max": settings_max},
    )

    mcp_calls = state.get("mcp_calls", []) or []

    from mcp_servers.registry import get_registry
    registry = get_registry()

    # Primary path: jobs MCP (visible in the UI). One call per country, merged.
    per_country = max(settings_max // len(countries_to_collect), 1)
    collected_postings = []
    collected_sources: list[str] = []
    for c in countries_to_collect:
        result = await registry.call_tool(
            "jobs",
            "search_jobs",
            {"role": role, "country": c, "max_results": per_country},
        )
        if result.success and result.result:
            try:
                partial = JobSearchResult.model_validate(result.result)
                collected_postings.extend(partial.postings)
                for src in partial.sources_queried:
                    if src not in collected_sources:
                        collected_sources.append(src)
                mcp_calls = [
                    *mcp_calls,
                    MCPCallRecord(
                        agent="market_collection",
                        server="jobs",
                        tool="search_jobs",
                        arguments={"role": role, "country": c, "max_results": per_country},
                        success=True,
                        summary=f"Collected {len(partial.postings)} postings for {c} "
                                f"({', '.join(partial.sources_queried) or 'no sources'})",
                    ).model_dump(),
                ]
            except Exception as e:
                logger.error(
                    "Could not validate jobs MCP result",
                    extra={"error": str(e), "country": c},
                )
                mcp_calls = [
                    *mcp_calls,
                    MCPCallRecord(
                        agent="market_collection",
                        server="jobs",
                        tool="search_jobs",
                        arguments={"role": role, "country": c, "max_results": per_country},
                        success=False,
                        summary=f"Failed for {c}: {str(e)[:80]}",
                    ).model_dump(),
                ]

    job_results: Optional[JobSearchResult] = None
    if collected_postings:
        job_results = JobSearchResult(
            query=role,
            postings=collected_postings,
            total_found=len(collected_postings),
            sources_queried=collected_sources,
            search_params={"role": role, "countries": countries_to_collect},
        )

    if job_results is None:
        # Fallback: call the underlying tool directly so we never hard-fail.
        from tools.job_search_tool import _search_adzuna, _search_arbeitnow
        logger.warning("Jobs MCP failed — falling back to direct tool calls")
        per_source = max(settings_max // (2 * len(countries_to_collect)), 1)
        adzuna_jobs = []
        for c in countries_to_collect:
            adzuna_jobs.extend(await _search_adzuna(
                role=role, country_code=c, max_results=per_source
            ))
        arbeitnow_jobs = await _search_arbeitnow(role=role, max_results=per_source)
        all_postings = adzuna_jobs + arbeitnow_jobs
        sources = []
        if adzuna_jobs:
            sources.append("adzuna")
        if arbeitnow_jobs:
            sources.append("arbeitnow")

        job_results = JobSearchResult(
            query=role,
            postings=all_postings,
            total_found=len(all_postings),
            sources_queried=sources,
            search_params={"role": role, "countries": countries_to_collect},
        )
        mcp_calls = [
            *mcp_calls,
            MCPCallRecord(
                agent="market_collection",
                server="jobs",
                tool="search_jobs",
                arguments={"role": role, "countries": countries_to_collect, "max_results": settings_max},
                success=len(all_postings) > 0,
                summary=(
                    f"Collected {len(all_postings)} postings (fallback path)"
                    if all_postings else "No postings found (fallback path)"
                ),
            ).model_dump(),
        ]

    postings_count = len(job_results.postings)
    sources_str = ", ".join(job_results.sources_queried) or "no sources configured"

    logger.info(
        "Market collection complete",
        extra={"total_postings": postings_count, "sources": job_results.sources_queried},
    )

    return {
        "job_search_results": job_results.model_dump(),
        "mcp_calls": mcp_calls,
        "completed_agents": state.get("completed_agents", []) + ["market_collection"],
        "messages": [
            AIMessage(
                content=(
                    f"🔍 Collected **{postings_count} job postings** for "
                    f"'{role}' from {sources_str}."
                )
            )
        ],
    }
