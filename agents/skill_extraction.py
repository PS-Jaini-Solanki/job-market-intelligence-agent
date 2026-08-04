"""
agents/skill_extraction.py
===========================
Skill Extraction Agent — extracts and counts skills from job postings.

RESPONSIBILITY:
- Run skill extraction on collected job postings
- Normalize skill names
- Compute frequency statistics
- Also parse salary data while we have the postings loaded

This agent reads job_search_results from state and writes
skill_statistics and salary_data back.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.job import JobSearchResult
from tools.salary_parser_tool import compute_salary_statistics
from tools.skill_extractor_tool import compute_skill_statistics, extract_skills_from_text

logger = logging.getLogger(__name__)


async def run_skill_extraction(state: GraphState) -> dict:
    """
    Skill Extraction agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with skill_statistics and salary_data.
    """
    job_results_raw = state.get("job_search_results")

    if not job_results_raw:
        logger.warning("No job search results available — skipping skill extraction")
        return {
            "skill_statistics": [],
            "errors": state.get("errors", []) + ["Skill extraction skipped: no job data"],
            "completed_agents": state.get("completed_agents", []) + ["skill_extraction"],
        }

    # Deserialize the job results
    job_results = JobSearchResult.model_validate(job_results_raw)
    role = state.get("role", "Software Engineer")
    country = state.get("country_filter")

    logger.info(
        "Skill extraction starting",
        extra={"postings": len(job_results.postings)},
    )

    # Step 1: Enrich postings with skills extracted from descriptions
    for posting in job_results.postings:
        if posting.description and len(posting.skills) == 0:
            posting.skills = extract_skills_from_text(posting.description)

    # Step 2: Compute skill statistics
    stats = compute_skill_statistics(job_results)

    # Step 3: Compute salary statistics
    salary_data = compute_salary_statistics(
        postings=job_results.postings,
        role=role,
        country=country,
    )

    # Serialize statistics for state storage
    serialized_stats = [
        {
            "skill_name": stat.skill.normalized_name,
            "category": stat.skill.category.value,
            "occurrence_count": stat.occurrence_count,
            "total_postings": stat.total_postings_analyzed,
            "percentage": stat.percentage,
            "sources": stat.sources,
            "evidence": stat.as_evidence_string(),
        }
        for stat in stats[:50]  # Store top 50 skills
    ]

    top_5_skills = ", ".join(
        s["skill_name"] for s in serialized_stats[:5]
    ) or "none found"

    logger.info(
        "Skill extraction complete",
        extra={
            "unique_skills": len(stats),
            "salary_sample": salary_data.sample_count,
        },
    )

    return {
        "skill_statistics": serialized_stats,
        "salary_data": salary_data.model_dump(),
        # Update job results with enriched skills
        "job_search_results": job_results.model_dump(),
        "completed_agents": state.get("completed_agents", []) + ["skill_extraction"],
        "messages": [
            AIMessage(
                content=(
                    f"🧠 Extracted **{len(stats)} unique skills** from "
                    f"{len(job_results.postings)} postings.\n"
                    f"Top skills: {top_5_skills}"
                )
            )
        ],
    }
