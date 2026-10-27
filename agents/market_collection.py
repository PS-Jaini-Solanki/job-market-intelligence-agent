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
    # each market. None means the user named no country.
    countries_to_collect: list[Optional[str]] = [
        c.upper() for c in countries if isinstance(c, str) and c
    ]
    if not countries_to_collect:
        countries_to_collect = [country.upper() if country else None]

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
    mcp_failed = False
    for c in countries_to_collect:
        arguments = {"role": role, "max_results": per_country}
        if c:
            arguments["country"] = c
        label = c or "no country specified"
        result = await registry.call_tool("jobs", "search_jobs", arguments)
        partial: Optional[JobSearchResult] = None
        error = result.error
        if result.success and result.result:
            try:
                partial = JobSearchResult.model_validate(result.result)
            except Exception as e:
                error = str(e)
        if partial is None:
            mcp_failed = True
            logger.error("Jobs MCP call failed", extra={"error": error, "country": label})
        else:
            collected_postings.extend(partial.postings)
            collected_sources.extend(s for s in partial.sources_queried if s not in collected_sources)
        mcp_calls = [
            *mcp_calls,
            MCPCallRecord(
                agent="market_collection",
                server="jobs",
                tool="search_jobs",
                arguments=arguments,
                success=partial is not None,
                summary=(
                    f"Collected {len(partial.postings)} postings for {label} "
                    f"({', '.join(partial.sources_queried) or 'no matching postings'})"
                    if partial is not None else f"Failed for {label}: {str(error)[:80]}"
                ),
            ).model_dump(),
        ]

    if mcp_failed and not collected_postings:
        # Fallback: call the collector directly so an MCP-layer failure
        # never hides data the job APIs can still provide.
        from tools.job_search_tool import collect_postings
        logger.warning("Jobs MCP failed — falling back to direct tool calls")
        for c in countries_to_collect:
            partial = await collect_postings(role, c, per_country)
            collected_postings.extend(partial.postings)
            collected_sources.extend(s for s in partial.sources_queried if s not in collected_sources)
        mcp_calls = [
            *mcp_calls,
            MCPCallRecord(
                agent="market_collection",
                server="jobs",
                tool="search_jobs",
                arguments={"role": role, "countries": countries_to_collect, "max_results": per_country},
                success=bool(collected_postings),
                summary=f"Collected {len(collected_postings)} postings (direct fallback path)",
            ).model_dump(),
        ]

    job_results = JobSearchResult(
        query=role,
        postings=collected_postings,
        total_found=len(collected_postings),
        sources_queried=collected_sources,
        search_params={"role": role, "countries": [c for c in countries_to_collect if c]},
    )

    postings_count = len(job_results.postings)
    sources_str = ", ".join(job_results.sources_queried) or "no source (no matching postings)"

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
