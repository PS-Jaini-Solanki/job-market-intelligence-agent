"""
agents/recommendation.py
=========================
Recommendation Agent — generates personalized learning roadmap.

RESPONSIBILITY:
- Look at skill gaps (from resume analysis OR just market demand)
- Look at technology trends (GitHub + job data)
- Generate prioritized, evidence-backed recommendations
- Use LLM only to format/explain — not to invent data

HALLUCINATION PREVENTION:
The LLM only sees real data from state. It formats recommendations
using that data. It cannot invent statistics.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState
from prompts.report_prompt import RECOMMENDATION_SYSTEM_PROMPT, build_recommendation_prompt
from services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


async def run_recommendation(state: GraphState) -> dict:
    """
    Recommendation agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with recommendations list.
    """
    logger.info("Recommendation agent starting")

    # Gather available evidence
    skill_stats = state.get("skill_statistics") or []
    resume_analysis = state.get("resume_analysis")
    trend_data = state.get("trend_data")
    role = state.get("role", "AI Engineer")
    country = state.get("country_filter", "global")

    # Build analysis data summary for the LLM
    analysis_summary: dict = {
        "role": role,
        "market": country,
        "top_demanded_skills": [
            f"{s['skill_name']}: {s['occurrence_count']}/{s['total_postings']} postings ({s['percentage']}%)"
            for s in skill_stats[:15]
        ],
    }

    if resume_analysis:
        missing = resume_analysis.get("missing_skills", [])
        high_priority_gaps = [
            m["skill"]["normalized_name"]
            for m in missing
            if m.get("priority") == "HIGH"
        ]
        analysis_summary["high_priority_gaps"] = high_priority_gaps
        analysis_summary["gap_score"] = resume_analysis.get("gap_score")

    if trend_data:
        ranked = sorted(
            trend_data.get("trends", []),
            key=lambda t: t.get("popularity_score", 0),
            reverse=True,
        )
        analysis_summary["top_technologies_by_trend"] = [
            f"{t['technology']}: score {t.get('popularity_score', 0):.1f}"
            for t in ranked[:10]
        ]

    # Use LLM to format recommendations from real data
    try:
        llm = get_llm_service()
        raw_recommendations = await llm.generate(
            user_prompt=build_recommendation_prompt(
                analysis_data=analysis_summary,
                user_context=f"Targeting {role} roles in {country}",
            ),
            system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
        )

        # Parse the LLM response into a list
        recommendations = _parse_recommendations(raw_recommendations)

    except Exception as e:
        logger.error("Recommendation LLM failed — using rule-based fallback", extra={"error": str(e)})
        recommendations = _rule_based_recommendations(skill_stats, resume_analysis)

    logger.info(
        "Recommendations generated",
        extra={"count": len(recommendations)},
    )

    return {
        "recommendations": recommendations,
        "completed_agents": state.get("completed_agents", []) + ["recommendation"],
        "messages": [
            AIMessage(
                content=f"💡 Generated **{len(recommendations)} recommendations**."
            )
        ],
    }


def _parse_recommendations(raw_text: str) -> list[str]:
    """
    Parse the LLM recommendation response into a list of strings.

    Args:
        raw_text: Raw LLM response text.

    Returns:
        List of recommendation strings.
    """
    lines = raw_text.strip().split("\n")
    recommendations = []

    for line in lines:
        line = line.strip()
        # Remove numbered list prefixes like "1.", "2.", "•", "-"
        import re
        clean = re.sub(r"^[\d]+\.\s*|^[-•*]\s*", "", line)
        if clean and len(clean) > 10:  # Skip very short lines
            recommendations.append(clean)

    return recommendations[:6]  # Max 6 recommendations


def _rule_based_recommendations(
    skill_stats: list[dict],
    resume_analysis: dict | None,
) -> list[str]:
    """
    Generate recommendations using rules when LLM is unavailable.

    Args:
        skill_stats: Skill frequency statistics from state.
        resume_analysis: Resume gap analysis, if available.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if resume_analysis:
        # Prioritize high-demand missing skills
        missing = resume_analysis.get("missing_skills", [])
        high_priority = [
            m for m in missing if m.get("priority") == "HIGH"
        ]
        for gap in high_priority[:3]:
            skill_name = gap["skill"]["normalized_name"]
            pct = gap.get("market_demand_percentage", 0)
            recommendations.append(
                f"Learn {skill_name} — required in {pct:.1f}% of analyzed job postings"
            )

    else:
        # Recommend top demanded skills
        for stat in skill_stats[:3]:
            recommendations.append(
                f"Strengthen {stat['skill_name']} skills — "
                f"in {stat['occurrence_count']}/{stat['total_postings']} postings ({stat['percentage']}%)"
            )

    if not recommendations:
        recommendations = [
            "Review the skills section of this report to identify high-demand technologies",
            "Focus on skills appearing in >30% of job postings in your target market",
        ]

    return recommendations
