"""
agents/trend_analysis.py
=========================
Trend Analysis Agent — combines GitHub data with job market data.

RESPONSIBILITY:
- Fetch GitHub statistics for technologies being analyzed
- Combine with job market skill demand data
- Compute popularity scores and rankings
- Store trend snapshot in state

This agent is only run when the query involves technology trend analysis
(e.g., "What's trending?", "Should I learn MCP or LangGraph?").
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState
from tools.github_trend_tool import analyze_github_trends
from tools.tech_popularity_tool import build_tech_trends
from models.trend import TrendSnapshot

logger = logging.getLogger(__name__)


async def run_trend_analysis(state: GraphState) -> dict:
    """
    Trend Analysis agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with trend_data.
    """
    technologies = state.get("technologies_to_analyze", [])

    if not technologies:
        logger.info("No technologies to analyze — skipping trend analysis")
        return {
            "completed_agents": state.get("completed_agents", []) + ["trend_analysis"],
            "messages": [AIMessage(content="📊 Trend analysis skipped — no technologies specified.")],
        }

    logger.info(
        "Trend analysis starting",
        extra={"technologies": technologies},
    )

    # Step 1: Fetch GitHub data for all technologies
    # analyze_github_trends is a LangChain tool — we call it directly
    github_raw = await analyze_github_trends.ainvoke({"technologies": technologies})

    # Step 2: Get job market counts from skill statistics
    skill_stats_raw = state.get("skill_statistics") or []
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
        github_data=github_raw,
        job_skill_stats=job_skill_counts,
        total_postings=total_postings,
        sources=["github", "adzuna", "arbeitnow"],
    )

    # Get top 5 for the status message
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
