"""
agents/trend_analysis.py
=========================
Trend Analysis Agent — combines GitHub data with job market data.

RESPONSIBILITY:
- Fetch GitHub statistics via the `github` MCP server (registry)
- Combine with job market skill demand data
- Compute popularity scores and rankings
- Store trend snapshot in state

Every GitHub call goes through the MCP registry so the UI can show
exactly which github tool was called for which technology.

This agent is only run when the query involves technology trend analysis
(e.g., "What's trending?", "Should I learn MCP or LangGraph?").
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.agent import MCPCallRecord
from models.trend import GitHubStats
from tools.github_trend_tool import TECH_TO_REPO
from tools.tech_popularity_tool import build_tech_trends

logger = logging.getLogger(__name__)

# How many of the postings' top skills to rank when the user names no technology.
TOP_SKILLS_WHEN_UNNAMED = 8


def _coerce_github_stats(raw: dict) -> GitHubStats | None:
    """
    Convert a github MCP `get_repo_stats` result into a GitHubStats model.

    Args:
        raw: Result dict from the github MCP server.

    Returns:
        GitHubStats instance, or None if the data isn't usable.
    """
    if not raw or "stars" not in raw or isinstance(raw.get("error"), str):
        return None
    try:
        return GitHubStats(**raw)
    except Exception as e:
        logger.debug("Could not coerce GitHub stats", extra={"error": str(e)})
        return None


async def run_trend_analysis(state: GraphState) -> dict:
    """
    Trend Analysis agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with trend_data.
    """
    technologies = state.get("technologies_to_analyze") or []
    skill_stats_raw = state.get("skill_statistics") or []

    # "What's trending for data engineers?" names no technology: rank the
    # skills the collected postings actually ask for, not a fixed list.
    if not technologies:
        technologies = [s["skill_name"] for s in skill_stats_raw[:TOP_SKILLS_WHEN_UNNAMED]]

    if not technologies:
        logger.info("No technologies to analyze — skipping trend analysis")
        return {
            "completed_agents": state.get("completed_agents", []) + ["trend_analysis"],
            "messages": [AIMessage(content="📊 Trend analysis skipped — no technologies named and no skills found in postings.")],
        }

    logger.info(
        "Trend analysis starting",
        extra={"technologies": technologies},
    )

    from mcp_servers.registry import get_registry
    registry = get_registry()
    mcp_calls = state.get("mcp_calls", []) or []
    github_data: dict[str, dict] = {}

    for tech in technologies:
        repo_path = TECH_TO_REPO.get(tech)

        if repo_path:
            # Canonical repo → get_repo_stats (detailed GitHub data).
            result = await registry.call_tool(
                "github", "get_repo_stats", {"repo_path": repo_path}
            )
            record = MCPCallRecord(
                agent="trend_analysis",
                server="github",
                tool="get_repo_stats",
                arguments={"repo_path": repo_path},
                success=result.success,
                summary=(
                    f"{tech}: {result.result.get('stars', 'n/a')}★"
                    if result.success and result.result
                    else (result.error or "failed")
                ),
            )
            mcp_calls = [*mcp_calls, record.model_dump()]

            if result.success and result.result and "error" not in result.result:
                github_data[tech] = result.result
                github_data[tech]["source"] = "github_api"
            else:
                github_data[tech] = {"error": "Could not fetch data"}
        else:
            # No canonical repo → get_trending topic search.
            result = await registry.call_tool(
                "github", "get_trending", {"topic": tech.lower().replace(" ", "-")}
            )
            record = MCPCallRecord(
                agent="trend_analysis",
                server="github",
                tool="get_trending",
                arguments={"topic": tech},
                success=result.success,
                summary=(
                    f"{tech}: {result.result.get('total_repos_with_topic', 'n/a')} repos"
                    if result.success and result.result
                    else (result.error or "failed")
                ),
            )
            mcp_calls = [*mcp_calls, record.model_dump()]

            if result.success and result.result:
                github_data[tech] = {
                    "technology": tech,
                    "topic_repo_count": result.result.get("total_repos_with_topic", 0),
                    "source": "github_topic_search",
                }
            else:
                github_data[tech] = {"error": result.error or "failed"}

    # Step 2: Get job market counts from skill statistics
    job_skill_counts: dict[str, int] = {
        stat["skill_name"]: stat["occurrence_count"]
        for stat in skill_stats_raw
    }
    total_postings = (
        skill_stats_raw[0]["total_postings"] if skill_stats_raw else 0
    )

    # Step 3: Build combined trend snapshot
    snapshot = build_tech_trends(
        technologies=technologies,
        github_data=github_data,
        job_skill_stats=job_skill_counts,
        total_postings=total_postings,
        sources=["github", *((state.get("job_search_results") or {}).get("sources_queried") or [])],
    )

    top_5 = snapshot.get_ranked(top_n=5)
    top_5_str = ", ".join(
        f"{t.technology} ({t.popularity_score:.0f}pts)" for t in top_5
    )

    logger.info(
        "Trend analysis complete",
        extra={"technologies_analyzed": len(snapshot.trends)},
    )

    return {
        "trend_data": snapshot.model_dump(),
        "mcp_calls": mcp_calls,
        "completed_agents": state.get("completed_agents", []) + ["trend_analysis"],
        "messages": [
            AIMessage(
                content=(
                    f"📈 Analyzed **{len(snapshot.trends)} technologies**.\n"
                    f"Top by score: {top_5_str or 'none'}"
                )
            )
        ],
    }
