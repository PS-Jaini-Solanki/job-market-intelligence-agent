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
import math
import re
from datetime import datetime
from typing import Optional

import httpx
from langchain_core.tools import tool

from config import get_settings
from models.job import JobPosting, JobSearchResult, SalaryRange

logger = logging.getLogger(__name__)

# Adzuna needs a country board; this one is used when the user names none.
DEFAULT_ADZUNA_COUNTRY = "US"

# Arbeitnow ignores its `search` parameter (it returns the same feed for any
# query), so we scan its newest pages and filter by title locally.
# ponytail: bounded scan of the newest pages; raise if recall is too low.
ARBEITNOW_MAX_PAGES = 3

# Title words treated as interchangeable when matching a role to a posting.
_ROLE_SUFFIX_EQUIVALENTS = {"engineer", "developer", "programmer"}

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

# Adzuna reports salaries in the local currency of each country board.
# We label them correctly so salary statistics are never misattributed.
ADZUNA_COUNTRY_CURRENCIES: dict[str, str] = {
    "IN": "INR",   # India — Rupee
    "US": "USD",   # United States — Dollar
    "GB": "GBP",   # United Kingdom — Pound
    "AU": "AUD",   # Australia — Dollar
    "CA": "CAD",   # Canada — Dollar
    "DE": "EUR",   # Germany — Euro
    "FR": "EUR",   # France — Euro
    "SG": "SGD",   # Singapore — Dollar
    "AE": "AED",   # UAE — Dirham
}

# Every country the job sources can serve, and how to show it to a user.
COUNTRY_NAMES: dict[str, str] = {
    "IN": "India",
    "US": "United States",
    "GB": "United Kingdom",
    "AU": "Australia",
    "CA": "Canada",
    "DE": "Germany",
    "FR": "France",
    "SG": "Singapore",
    "AE": "United Arab Emirates",
}
SUPPORTED_COUNTRIES: frozenset[str] = frozenset(ADZUNA_COUNTRY_PATHS)


def country_name(code: Optional[str]) -> str:
    """Human-readable name for an ISO code ('IN' -> 'India')."""
    if not code:
        return ""
    return COUNTRY_NAMES.get(code.upper(), code.upper())


def is_relevant_title(title: str, role: str) -> bool:
    """
    True if a posting title is about the requested role.

    Every word of the role must appear in the title, except that
    engineer/developer/programmer count as the same word. Precision beats
    recall here: an unrelated posting silently skews every percentage.
    """
    title_tokens = set(re.findall(r"[a-z0-9+#]+", title.lower()))
    for token in re.findall(r"[a-z0-9+#]+", role.lower()):
        if token in _ROLE_SUFFIX_EQUIVALENTS:
            if not title_tokens & _ROLE_SUFFIX_EQUIVALENTS:
                return False
        elif token not in title_tokens:
            return False
    return True


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

    if max_results < 1:
        return []

    country_path = ADZUNA_COUNTRY_PATHS.get(country_code.upper())
    if not country_path:
        logger.warning("Adzuna has no board for this country", extra={"country": country_code})
        return []
    # Adzuna serves at most 50 results per page.
    results_per_page = min(50, max_results)
    pages = math.ceil(max_results / results_per_page)

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
                        currency=ADZUNA_COUNTRY_CURRENCIES.get(
                            country_code.upper(), "USD"
                        ),
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
                if len(all_postings) >= max_results:
                    break

            logger.debug(
                "Adzuna page fetched",
                extra={"page": page, "results": len(results)},
            )
            if len(all_postings) >= max_results:
                break

    logger.info(
        "Adzuna search complete",
        extra={"role": role, "country": country_code, "count": len(all_postings)},
    )
    return all_postings


async def _search_arbeitnow(
    role: str,
    max_results: int,
    country_code: Optional[str] = None,
) -> list[JobPosting]:
    """
    Search jobs using the Arbeitnow API (no API key required).

    The API ignores its `search` parameter, so postings are kept only when
    the title matches the role and, if a country is given, the location is
    in that country.

    Args:
        role: Job title to search for.
        max_results: Maximum number of results to return.
        country_code: Optional ISO code; postings elsewhere are dropped.

    Returns:
        List of relevant JobPosting objects.
    """
    if max_results < 1:
        return []

    wanted_country = country_code.upper() if country_code else None
    all_postings: list[JobPosting] = []
    scanned = 0

    async with httpx.AsyncClient(timeout=get_settings().API_TIMEOUT) as client:
        for page in range(1, ARBEITNOW_MAX_PAGES + 1):
            url = "https://www.arbeitnow.com/api/job-board-api"
            try:
                response = await client.get(url, params={"search": role, "page": page})
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.error("Arbeitnow API error", extra={"error": str(e)})
                break

            jobs = data.get("data", [])
            if not jobs:
                break

            for item in jobs:
                scanned += 1
                title = item.get("title") or ""
                if not is_relevant_title(title, role):
                    continue
                location = item.get("location") or ""
                # Unknown/remote locations stay None and are excluded whenever
                # a specific country was requested.
                posting_country = _infer_country_from_location(location)
                if wanted_country and posting_country != wanted_country:
                    continue
                all_postings.append(JobPosting(
                    title=title,
                    company=item.get("company_name", "Unknown"),
                    location=location or "Remote",
                    country=posting_country,
                    description=item.get("description", ""),
                    salary=None,  # Arbeitnow rarely includes salary
                    skills=item.get("tags", []),  # vocabulary-filtered later
                    source="arbeitnow",
                    source_url=item.get("url"),
                    posted_at=_parse_date(item.get("created_at")),
                ))
                if len(all_postings) >= max_results:
                    break
            if len(all_postings) >= max_results:
                break

    logger.info(
        "Arbeitnow search complete",
        extra={"role": role, "country": wanted_country, "scanned": scanned,
               "relevant": len(all_postings)},
    )
    return all_postings


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

    # Arbeitnow sometimes emits epoch timestamps (int/float) for created_at.
    if isinstance(date_str, (int, float)):
        try:
            return datetime.fromtimestamp(float(date_str))
        except (ValueError, OverflowError, OSError):
            return None

    # Normalize the common ISO-8601 "Z" suffix to an explicit offset.
    normalized = str(date_str).strip().replace("Z", "+00:00")

    for fmt in (
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, fmt)
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
        "FR": ["france", "paris", "lyon"],
        "AE": ["united arab emirates", "dubai", "abu dhabi"],
    }
    # Whole-word match: a substring test tagged e.g. "Neukölln" as UK.
    for code, keywords in country_keywords.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", location_lower) for kw in keywords):
            return code
    return None


async def collect_postings(
    role: str,
    country: Optional[str] = None,
    max_results: int = 100,
) -> JobSearchResult:
    """
    Collect postings for one role in one market from every source.

    Adzuna (country-specific, relevant results) gets the full budget;
    Arbeitnow fills whatever budget is left with postings that pass the
    title/country filter. With no country, Adzuna uses its default board
    and Arbeitnow is not filtered by country.

    Args:
        role: Job title to search for.
        country: ISO country code, or None if the user named none.
        max_results: Maximum total postings.

    Returns:
        JobSearchResult with postings and the sources that returned data.
    """
    adzuna_country = (country or DEFAULT_ADZUNA_COUNTRY).upper()
    adzuna_jobs = await _search_adzuna(role, adzuna_country, max_results)
    arbeitnow_jobs = await _search_arbeitnow(
        role, max_results - len(adzuna_jobs), country_code=country
    )

    sources = [
        name for name, jobs in (("adzuna", adzuna_jobs), ("arbeitnow", arbeitnow_jobs)) if jobs
    ]
    result = JobSearchResult(
        query=role,
        postings=adzuna_jobs + arbeitnow_jobs,
        total_found=len(adzuna_jobs) + len(arbeitnow_jobs),
        sources_queried=sources,
        search_params={"role": role, "country": country},
    )
    logger.info(
        "Job search complete",
        extra={"role": role, "country": country, "total_found": result.total_found,
               "sources": sources},
    )
    return result


@tool
async def search_jobs(
    role: str,
    country: Optional[str] = None,
    max_results: int = 100,
) -> dict:
    """
    Search for job postings from multiple sources.

    This is a LangChain tool that agents can call.
    Returns a serialized JobSearchResult as a dictionary.

    Args:
        role: Job title to search for (e.g., 'AI Engineer', 'Data Scientist').
        country: ISO country code (e.g., 'IN', 'US', 'GB'), or None.
        max_results: Maximum total postings to collect. Default: 100.

    Returns:
        Serialized JobSearchResult containing all collected postings.
    """
    # Return as dict because LangChain tools must return serializable types
    return (await collect_postings(role, country, max_results)).model_dump()
