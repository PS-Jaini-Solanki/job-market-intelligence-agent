"""
tools/tech_popularity_tool.py
==============================
Combines GitHub and job market data to rank technology popularity.

This tool aggregates signals from:
1. GitHub statistics (stars, forks, recency)
2. Job market demand (from job search results)
3. Optional: Reddit discussion count

The resulting popularity_score is a transparent, computed metric —
not an LLM opinion.

See TechTrend.popularity_score in models/trend.py for the formula.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from models.trend import GitHubStats, TechTrend, TrendSnapshot

logger = logging.getLogger(__name__)


def build_tech_trends(
    technologies: list[str],
    github_data: dict[str, dict],
    job_skill_stats: dict[str, int],
    total_postings: int,
    sources: list[str],
) -> TrendSnapshot:
    """
    Combine GitHub and job data into a TrendSnapshot.

    Args:
        technologies: List of technology names to include.
        github_data: Output from analyze_github_trends tool.
        job_skill_stats: Dict of {skill_name: occurrence_count} from extract_skills.
        total_postings: Total job postings analyzed.
        sources: Data source labels.

    Returns:
        TrendSnapshot with a TechTrend for each technology.
    """
    trends: list[TechTrend] = []

    for tech in technologies:
        # Get GitHub data for this technology
        gh_raw = github_data.get(tech, {})
        github_stats: Optional[GitHubStats] = None

        if gh_raw and "stars" in gh_raw:
            try:
                github_stats = GitHubStats(**gh_raw)
            except Exception as e:
                logger.debug("Could not parse GitHub stats", extra={"tech": tech, "error": str(e)})

        # Get job market count
        job_count = job_skill_stats.get(tech, 0)

        trend = TechTrend(
            technology=tech,
            github_stats=github_stats,
            job_count=job_count,
            total_postings_analyzed=total_postings,
            sources=sources,
        )
        trends.append(trend)

    snapshot = TrendSnapshot(
        trends=trends,
        query_context=f"Analysis of {len(technologies)} technologies",
    )

    logger.info(
        "Technology trends computed",
        extra={
            "technologies": len(technologies),
            "with_github": sum(1 for t in trends if t.github_stats),
        },
    )
    return snapshot


@tool
def rank_technologies(
    technologies: list[str],
    github_data: dict,
    skill_statistics: dict,
) -> dict:
    """
    Rank technologies by a composite popularity score.

    Combines GitHub activity and job market demand into a
    transparent, computed score (not an LLM opinion).

    Args:
        technologies: List of technology names.
        github_data: Output from analyze_github_trends tool.
        skill_statistics: Output from extract_skills tool.

    Returns:
        Ranked list of technologies with scores and evidence.
    """
    # Extract job counts from skill statistics
    job_skill_stats: dict[str, int] = {}
    total_postings = skill_statistics.get("total_postings_analyzed", 0)
    for stat in skill_statistics.get("statistics", []):
        job_skill_stats[stat["skill_name"]] = stat["occurrence_count"]

    all_sources: list[str] = ["github", *skill_statistics.get("sources", [])]

    snapshot = build_tech_trends(
        technologies=technologies,
        github_data=github_data,
        job_skill_stats=job_skill_stats,
        total_postings=total_postings,
        sources=all_sources,
    )

    ranked = snapshot.get_ranked(top_n=len(technologies))

    return {
        "query_context": snapshot.query_context,
        "total_postings_analyzed": total_postings,
        "ranked_technologies": [
            {
                "rank": i + 1,
                "technology": t.technology,
                "popularity_score": t.popularity_score,
                "github_stars": t.github_stats.stars if t.github_stats else None,
                "job_count": t.job_count,
                "job_demand_percentage": t.job_demand_percentage,
                "evidence": t.as_evidence_string(),
            }
            for i, t in enumerate(ranked)
        ],
        "score_explanation": (
            "Popularity score = GitHub stars (log-scaled, 0–50pts) + "
            "Job demand % (0–50pts). Max score = 100."
        ),
    }


@tool
def compare_technologies(
    technology_a: str,
    technology_b: str,
    github_data: dict,
    skill_statistics: dict,
) -> dict:
    """
    Compare two technologies head-to-head.

    Provides a factual comparison based on:
    - GitHub stars
    - Job market demand
    - Composite popularity score

    Args:
        technology_a: First technology name.
        technology_b: Second technology name.
        github_data: Output from analyze_github_trends tool.
        skill_statistics: Output from extract_skills tool.

    Returns:
        Head-to-head comparison dictionary.
    """
    job_skill_stats: dict[str, int] = {}
    total_postings = skill_statistics.get("total_postings_analyzed", 0)
    for stat in skill_statistics.get("statistics", []):
        job_skill_stats[stat["skill_name"]] = stat["occurrence_count"]

    techs = [technology_a, technology_b]
    snapshot = build_tech_trends(
        technologies=techs,
        github_data=github_data,
        job_skill_stats=job_skill_stats,
        total_postings=total_postings,
        sources=["github", "job_market"],
    )

    trend_a = snapshot.get_by_technology(technology_a)
    trend_b = snapshot.get_by_technology(technology_b)

    def trend_to_dict(trend: Optional[TechTrend]) -> dict:
        if not trend:
            return {"error": "No data available"}
        return {
            "technology": trend.technology,
            "popularity_score": trend.popularity_score,
            "github_stars": trend.github_stats.stars if trend.github_stats else None,
            "job_count": trend.job_count,
            "job_demand_percentage": trend.job_demand_percentage,
            "evidence": trend.as_evidence_string(),
        }

    winner = None
    if trend_a and trend_b:
        if trend_a.popularity_score > trend_b.popularity_score:
            winner = technology_a
        elif trend_b.popularity_score > trend_a.popularity_score:
            winner = technology_b
        else:
            winner = "tie"

    return {
        technology_a: trend_to_dict(trend_a),
        technology_b: trend_to_dict(trend_b),
        "winner_by_score": winner,
        "data_note": (
            "Winner is determined by composite score only. "
            "Consider your specific use case when choosing."
        ),
    }
