"""
agents/country_comparison.py
==============================
Country Comparison Agent — compares job markets across countries.

RESPONSIBILITY:
- Read the collected job postings from state
- Call the `jobs` MCP `compare_countries` tool
- Store the cross-country comparison in state for the report

Only runs when the planner includes a compare_countries task
(e.g., "Compare AI jobs in India vs USA").
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.agent import MCPCallRecord

logger = logging.getLogger(__name__)

# Common ISO codes for ad-hoc comparisons when the query names no countries.
_COUNTRY_CODES = ["IN", "US", "GB", "DE", "CA", "FR", "AU", "SG", "AE"]


def _country_codes_from_plan(state: GraphState) -> list[str]:
    """
    Extract country codes from the plan's compare_countries task.

    Args:
        state: Current graph state.

    Returns:
        List of ISO country codes.
    """
    plan = state.get("execution_plan", {})
    for task in plan.get("tasks", []):
        if task.get("task_type") == "compare_countries":
            params = task.get("parameters", {})
            codes = params.get("country_codes") or params.get("countries")
            if isinstance(codes, list) and codes:
                return [c.upper() for c in codes if isinstance(c, str)]
    return []


async def run_country_comparison(state: GraphState) -> dict:
    """
    Country Comparison agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with country_comparison populated.
    """
    job_results_raw = state.get("job_search_results")
    role = state.get("role", "Software Engineer")

    if not job_results_raw:
        logger.warning("No job data available for country comparison")
        return {
            "country_comparison": {"error": "No job data available for comparison"},
            "completed_agents": state.get("completed_agents", []) + ["comparison_agent"],
            "messages": [AIMessage(content="🌍 Country comparison skipped — no job data.")],
        }

    country_codes = _country_codes_from_plan(state)
    # Prefer the countries the user actually named (stored by the classifier).
    if not country_codes:
        country_codes = state.get("countries") or []
    # Last resort: compare the extracted country against the largest markets.
    if not country_codes:
        base = state.get("country_filter")
        country_codes = [base, *[c for c in _COUNTRY_CODES if c != base]][:3]

    logger.info(
        "Country comparison starting",
        extra={"role": role, "countries": country_codes},
    )

    mcp_calls = state.get("mcp_calls", []) or []
    try:
        from mcp_servers.registry import get_registry
        registry = get_registry()
        result = await registry.call_tool(
            "jobs",
            "compare_countries",
            {
                "role": role,
                "country_codes": country_codes,
            },
        )

        if result.success and result.result:
            mcp_calls = [
                *mcp_calls,
                MCPCallRecord(
                    agent="comparison_agent",
                    server="jobs",
                    tool="compare_countries",
                    arguments={"role": role, "country_codes": country_codes},
                    success=True,
                    summary=f"Compared {len(country_codes)} countries for '{role}'",
                ).model_dump(),
            ]
            comparison = result.result
        else:
            mcp_calls = [
                *mcp_calls,
                MCPCallRecord(
                    agent="comparison_agent",
                    server="jobs",
                    tool="compare_countries",
                    arguments={"role": role, "country_codes": country_codes},
                    success=False,
                    summary=result.error or "comparison failed",
                ).model_dump(),
            ]
            comparison = {"error": result.error or "comparison failed"}
    except Exception as e:
        logger.error("Country comparison failed", extra={"error": str(e)})
        comparison = {"error": str(e)}

    return {
        "country_comparison": comparison,
        "mcp_calls": mcp_calls,
        "completed_agents": state.get("completed_agents", []) + ["comparison_agent"],
        "messages": [
            AIMessage(
                content=f"🌍 Compared **{', '.join(country_codes)}** markets for `{role}`."
            )
        ],
    }
