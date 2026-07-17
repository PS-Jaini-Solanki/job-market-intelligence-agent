"""
tools/job_search_tool.py
========================
Collects real job postings from multiple free APIs.

DATA SOURCES:
1. Adzuna API — https://developer.adzuna.com (free developer account)
2. Arbeitnow API — https://www.arbeitnow.com/api (completely free, no key needed)

HALLUCINATION PREVENTION:
- This tool is the ONLY source of job market facts.
- Every JobPosting includes a source field and source_url.
- The LLM interprets this data — it never generates its own "facts".

HOW TO USE:
    result = await search_jobs(role="AI Engineer", country="IN", max_results=100)
    print(f"Found {result.count} jobs from {result.sources_queried}")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from models.job import JobPosting, JobSearchResult, SalaryRange

logger = logging.getLogger(__name__)

# Country code mapping for Adzuna API
# Adzuna uses different URL paths for different countries
ADZUNA_COUNTRY_PATHS: dict[str, str] = {
    "IN": "in",    # India
    "US": "us",    # United States
    "GB": "gb",    # United Kingdom
    "AU": "au",    # Australia
    "CA": "ca",    # Canada
    "DE": "de",    # Germany
    "FR": "fr",    # France
    "SG": "sg",    # Singapore
    "AE": "ae",    # UAE
}


async def _search_adzuna(
    role: str,
    country_code: str,
    max_results: int,
) -> list[JobPosting]:
    """
    Search jobs using the Adzuna API.

    Args:
        role: Job title to search for.
        country_code: ISO country code (e.g., 'IN', 'US').
        max_results: Maximum number of results to fetch.

    Returns:
        List of JobPosting objects, or empty list on failure.
    """
    settings = get_settings()

    if not settings.has_adzuna:
        logger.warning("Adzuna credentials not configured — skipping Adzuna search")
        return []

    country_path = ADZUNA_COUNTRY_PATHS.get(country_code.upper(), "us")
    # Calculate pages needed (Adzuna returns 10 results per page by default)
    results_per_page = min(50, max_results)
    pages = max(1, max_results // results_per_page)

    all_postings: list[JobPosting] = []

    async with httpx.AsyncClient(timeout=get_settings().API_TIMEOUT) as client:
        for page in range(1, pages + 1):
            url = (
                f"https://api.adzuna.com/v1/api/jobs/{country_path}/search/{page}"
            )
            params = {
                "app_id": settings.ADZUNA_APP_ID,
                "app_key": settings.ADZUNA_APP_KEY,
                "what": role,
                "results_per_page": results_per_page,
                "content-type": "application/json",
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as e:
                logger.error("Adzuna API error", extra={"error": str(e), "page": page})
                break

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                # Parse salary if available
                salary = None
                if item.get("salary_min") or item.get("salary_max"):
                    salary = SalaryRange(
                        min_value=item.get("salary_min"),
                        max_value=item.get("salary_max"),
                        currency="GBP" if country_path == "gb" else "USD",
                        period="year",
                    )

                posting = JobPosting(
                    title=item.get("title", "Unknown Title"),
                    company=item.get("company", {}).get("display_name", "Unknown"),
                    location=item.get("location", {}).get("display_name", "Unknown"),
                    country=country_code.upper(),
                    description=item.get("description", ""),
                    salary=salary,
                    skills=[],  # Skills will be extracted by SkillExtractionAgent
                    source="adzuna",
                    source_url=item.get("redirect_url"),
                    posted_at=_parse_date(item.get("created")),
                )
                all_postings.append(posting)

            logger.debug(
                "Adzuna page fetched",
                extra={"page": page, "results": len(results)},
            )

    logger.info(
        "Adzuna search complete",
        extra={"role": role, "country": country_code, "count": len(all_postings)},
    )
    return all_postings


async def _search_arbeitnow(
    role: str,
    max_results: int,
) -> list[JobPosting]:
    """
    Search jobs using the Arbeitnow API (no API key required).

    Arbeitnow focuses on tech jobs and is completely free.

    Args:
        role: Job title to search for.
        max_results: Maximum number of results to fetch.

    Returns:
        List of JobPosting objects.
    """
    all_postings: list[JobPosting] = []

    async with httpx.AsyncClient(timeout=get_settings().API_TIMEOUT) as client:
        page = 1
        while len(all_postings) < max_results:
            url = "https://www.arbeitnow.com/api/job-board-api"
            params = {
                "search": role,
                "page": page,
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as e:
                logger.error("Arbeitnow API error", extra={"error": str(e)})
                break

            jobs = data.get("data", [])
            if not jobs:
                break

            for item in jobs:
                # Arbeitnow jobs are mostly remote/European
                posting = JobPosting(
                    title=item.get("title", "Unknown Title"),
                    company=item.get("company_name", "Unknown"),
                    location=item.get("location", "Remote"),
                    country=_infer_country_from_location(item.get("location", "")),
                    description=item.get("description", ""),
                    salary=None,  # Arbeitnow rarely includes salary
                    skills=item.get("tags", []),  # Tags are a good skill proxy
                    source="arbeitnow",
                    source_url=item.get("url"),
                    posted_at=_parse_date(item.get("created_at")),
                )
                all_postings.append(posting)
                if len(all_postings) >= max_results:
                    break

            page += 1

    logger.info(
        "Arbeitnow search complete",
        extra={"role": role, "count": len(all_postings)},
    )
    return all_postings[:max_results]


def _parse_date(date_str: str | None) -> datetime | None:
    """
    Parse a date string from API responses.

    Args:
        date_str: Date string in various formats.

    Returns:
        datetime object, or None if parsing fails.
    """
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S+00:00"):
        try:
            return datetime.strptime(date_str[:19], fmt[:len(date_str)])
        except (ValueError, TypeError):
            continue
    return None


def _infer_country_from_location(location: str) -> str | None:
    """
    Try to infer a country code from a location string.

    This is a simple heuristic — not 100% accurate.

    Args:
        location: Location string from a job posting.

    Returns:
        ISO country code string, or None if unknown.
    """
    location_lower = location.lower()
    country_keywords = {
        "IN": ["india", "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "pune", "chennai"],
        "US": ["united states", "usa", "new york", "san francisco", "seattle", "austin", "remote us"],
        "GB": ["united kingdom", "uk", "london", "manchester", "edinburgh"],
        "AU": ["australia", "sydney", "melbourne", "brisbane"],
        "DE": ["germany", "berlin", "munich", "hamburg"],
        "CA": ["canada", "toronto", "vancouver", "montreal"],
        "SG": ["singapore"],
    }
    for code, keywords in country_keywords.items():
        if any(kw in location_lower for kw in keywords):
            return code
    return None


@tool
async def search_jobs(
    role: str,
    country: str = "US",
    max_results: int = 100,
) -> dict:
    """
    Search for job postings from multiple sources.

    This is a LangChain tool that agents can call.
    Returns a serialized JobSearchResult as a dictionary.

    Args:
        role: Job title to search for (e.g., 'AI Engineer', 'Data Scientist').
        country: ISO country code (e.g., 'IN', 'US', 'GB'). Default: 'US'.
        max_results: Maximum total postings to collect. Default: 100.

    Returns:
        Serialized JobSearchResult containing all collected postings.
    """
    logger.info(
        "Job search started",
        extra={"role": role, "country": country, "max_results": max_results},
    )

    per_source = max_results // 2  # Split budget evenly between sources

    # Collect from both sources in parallel would be ideal, but we do sequential
    # here to keep the code simpler and avoid rate limit issues
    adzuna_jobs = await _search_adzuna(
        role=role,
        country_code=country,
        max_results=per_source,
    )

    arbeitnow_jobs = await _search_arbeitnow(
        role=role,
        max_results=per_source,
    )

    all_postings = adzuna_jobs + arbeitnow_jobs
    sources_queried = []
    if adzuna_jobs:
        sources_queried.append("adzuna")
    if arbeitnow_jobs:
        sources_queried.append("arbeitnow")

    result = JobSearchResult(
        query=role,
        postings=all_postings,
        total_found=len(all_postings),
        sources_queried=sources_queried,
        search_params={"role": role, "country": country},
    )

    logger.info(
        "Job search complete",
        extra={
            "total_found": result.total_found,
            "sources": sources_queried,
        },
    )

    # Return as dict because LangChain tools must return serializable types
    return result.model_dump()
