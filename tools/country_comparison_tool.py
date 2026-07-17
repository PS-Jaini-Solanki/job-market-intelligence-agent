"""
tools/country_comparison_tool.py
=================================
Compares job market conditions across different countries.

This tool takes job search results and produces a side-by-side
comparison of skill demand, salary ranges, and job availability
for different countries.

All comparisons are based on actual collected job data.
If a country has too few postings, we flag it as unreliable.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from models.job import JobSearchResult
from tools.salary_parser_tool import compute_salary_statistics
from tools.skill_extractor_tool import compute_skill_statistics

logger = logging.getLogger(__name__)

# Minimum postings for a country to be considered statistically meaningful
MIN_POSTINGS_FOR_COMPARISON = 10


@tool
def compare_countries(
    job_search_result: dict,
    country_codes: list[str],
    top_skills: int = 10,
) -> dict:
    """
    Compare job market data across multiple countries.

    For each country, computes:
    - Number of job postings found
    - Top required skills (with counts and percentages)
    - Salary data (where available)

    Args:
        job_search_result: Serialized JobSearchResult from search_jobs tool.
        country_codes: List of ISO country codes to compare (e.g., ['IN', 'US']).
        top_skills: Number of top skills to include per country.

    Returns:
        Dictionary with per-country analysis and a comparison table.
    """
    result = JobSearchResult.model_validate(job_search_result)
    role = result.search_params.get("role", "Unknown Role")

    comparison: dict[str, dict] = {}

    for country_code in country_codes:
        country_upper = country_code.upper()

        # Filter postings for this country
        country_postings = result.get_postings_by_country(country_upper)
        posting_count = len(country_postings)

        logger.info(
            "Analyzing country",
            extra={"country": country_upper, "postings": posting_count},
        )

        if posting_count < MIN_POSTINGS_FOR_COMPARISON:
            comparison[country_upper] = {
                "posting_count": posting_count,
                "status": "insufficient_data",
                "warning": (
                    f"Only {posting_count} postings found for {country_upper}. "
                    f"Minimum {MIN_POSTINGS_FOR_COMPARISON} required for reliable analysis."
                ),
            }
            continue

        # Build a mini JobSearchResult for this country's postings
        from models.job import JobSearchResult as JSR
        country_result = JSR(
            query=result.query,
            postings=country_postings,
            total_found=posting_count,
            sources_queried=result.sources_queried,
            search_params={"role": role, "country": country_upper},
        )

        # Compute skills for this country
        skill_stats = compute_skill_statistics(country_result)
        top_skill_stats = skill_stats[:top_skills]

        # Compute salary for this country
        salary_data = compute_salary_statistics(
            postings=country_postings,
            role=role,
            country=country_upper,
        )

        comparison[country_upper] = {
            "posting_count": posting_count,
            "status": "ok",
            "top_skills": [
                {
                    "skill": stat.skill.normalized_name,
                    "occurrences": stat.occurrence_count,
                    "percentage": stat.percentage,
                    "evidence": stat.as_evidence_string(),
                }
                for stat in top_skill_stats
            ],
            "salary": {
                "min": salary_data.min_salary,
                "max": salary_data.max_salary,
                "median": salary_data.median_salary,
                "currency": salary_data.currency,
                "sample_count": salary_data.sample_count,
                "is_reliable": salary_data.sample_count >= 5,
            },
        }

    # Identify common and unique skills across countries
    countries_with_data = [c for c, v in comparison.items() if v.get("status") == "ok"]

    common_skills = _find_common_skills(comparison, countries_with_data)
    unique_skills = _find_unique_skills(comparison, countries_with_data)

    return {
        "role_analyzed": role,
        "countries": comparison,
        "cross_country_insights": {
            "common_skills": common_skills,
            "unique_skills_by_country": unique_skills,
        },
        "sources": result.sources_queried,
        "data_note": (
            "Country comparison is based on job postings collected from public APIs. "
            "The distribution of postings per country reflects the sources' coverage, "
            "not necessarily the true market size."
        ),
    }


def _find_common_skills(
    comparison: dict[str, dict],
    countries: list[str],
) -> list[str]:
    """
    Find skills that appear in the top 10 for ALL analyzed countries.

    Args:
        comparison: The country comparison dictionary.
        countries: List of country codes with sufficient data.

    Returns:
        List of skill names that are universally demanded.
    """
    if not countries:
        return []

    # Get top skill names for each country
    country_skill_sets: list[set[str]] = []
    for country in countries:
        skills = comparison.get(country, {}).get("top_skills", [])
        country_skill_sets.append({s["skill"] for s in skills})

    if not country_skill_sets:
        return []

    # Intersection of all sets = common to all countries
    common = country_skill_sets[0].copy()
    for skill_set in country_skill_sets[1:]:
        common &= skill_set

    return sorted(common)


def _find_unique_skills(
    comparison: dict[str, dict],
    countries: list[str],
) -> dict[str, list[str]]:
    """
    Find skills that appear in the top 10 for ONE country but not others.

    Args:
        comparison: The country comparison dictionary.
        countries: List of country codes with sufficient data.

    Returns:
        Dict of {country_code: [unique_skill_names]}.
    """
    if not countries:
        return {}

    # Build skill sets per country
    country_skills: dict[str, set[str]] = {}
    for country in countries:
        skills = comparison.get(country, {}).get("top_skills", [])
        country_skills[country] = {s["skill"] for s in skills}

    # All skills combined
    all_skills: set[str] = set()
    for skills in country_skills.values():
        all_skills |= skills

    unique: dict[str, list[str]] = {}
    for country in countries:
        my_skills = country_skills[country]
        other_skills: set[str] = set()
        for other_country, skills in country_skills.items():
            if other_country != country:
                other_skills |= skills

        unique[country] = sorted(my_skills - other_skills)

    return unique
