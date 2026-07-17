"""
tools/salary_parser_tool.py
============================
Parses salary information from job posting data.

HALLUCINATION PREVENTION:
This tool ONLY extracts salary data that actually exists in job postings.
It NEVER generates, estimates, or infers salary ranges.

If insufficient postings have salary data, it returns:
  "Insufficient evidence available."
instead of making up a number.

WHY SALARY DATA IS SPARSE:
Many companies don't list salaries publicly (especially in some markets
like India where it's culturally less common). We acknowledge this
limitation honestly rather than filling gaps with estimates.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.tools import tool

from models.job import JobPosting, JobSearchResult, SalaryData, SalaryRange

logger = logging.getLogger(__name__)

# Minimum sample size for reliable salary statistics
# Below this, we flag the data as potentially unreliable
MIN_RELIABLE_SAMPLE = 5

# Currency patterns for extracting salary from raw text
# These handle formats like: "$120,000", "£80k", "INR 15 LPA", "€90K", "15 LPA"
SALARY_TEXT_PATTERNS: list[tuple[str, str]] = [
    # "15 LPA" or "15-20 LPA" (Indian format: Lakhs Per Annum)
    (r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*LPA", "INR_LPA"),
    # "$120,000" or "$120k" or "$120K"
    (r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*[kK]?", "USD"),
    # "£80,000" or "£80k"
    (r"£\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*[kK]?", "GBP"),
    # "€90,000"
    (r"€\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*[kK]?", "EUR"),
    # "INR 1500000" or "Rs. 15,00,000"
    (r"(?:INR|Rs\.?)\s*([\d,]+)", "INR"),
]


def parse_salary_from_text(text: str) -> Optional[SalaryRange]:
    """
    Try to extract a salary range from arbitrary text.

    Args:
        text: Text that might contain salary information.

    Returns:
        SalaryRange if found, None otherwise.
    """
    if not text:
        return None

    for pattern, currency in SALARY_TEXT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                min_val = float(match.group(1).replace(",", ""))
                max_val = (
                    float(match.group(2).replace(",", ""))
                    if match.lastindex >= 2 and match.group(2)
                    else min_val
                )

                # Handle "k" suffix (e.g., $120k → $120,000)
                if "k" in text[match.start():match.end()].lower():
                    min_val *= 1000
                    max_val *= 1000

                # Handle LPA conversion (1 LPA = 100,000 INR)
                if currency == "INR_LPA":
                    min_val *= 100_000
                    max_val *= 100_000
                    currency = "INR"

                return SalaryRange(
                    min_value=min_val,
                    max_value=max_val,
                    currency=currency,
                    period="year",
                )
            except (ValueError, IndexError):
                continue

    return None


def compute_salary_statistics(
    postings: list[JobPosting],
    role: str,
    country: Optional[str] = None,
) -> SalaryData:
    """
    Compute salary statistics from a list of job postings.

    Only considers postings that explicitly list a salary.
    Never estimates or fills in missing data.

    Args:
        postings: List of job postings to analyze.
        role: The job role being analyzed (for labeling).
        country: Optional country filter (ISO code).

    Returns:
        SalaryData with honest statistics and sample count.
    """
    # Filter by country if specified
    if country:
        filtered = [p for p in postings if p.country == country.upper()]
    else:
        filtered = postings

    # Try to enrich salary from description text if not already parsed
    for posting in filtered:
        if posting.salary is None and posting.description:
            posting.salary = parse_salary_from_text(posting.description)

    # Collect all salaries (in original currency — we report as-is)
    # Group by currency for separate statistics
    salary_values_by_currency: dict[str, list[float]] = {}
    sources_seen: set[str] = set()

    for posting in filtered:
        if posting.salary:
            sal = posting.salary
            if sal.min_value or sal.max_value:
                # Use midpoint for statistics
                midpoint = (
                    (sal.min_value + sal.max_value) / 2
                    if (sal.min_value and sal.max_value)
                    else (sal.min_value or sal.max_value)
                )
                if midpoint:
                    bucket = salary_values_by_currency.setdefault(sal.currency, [])
                    bucket.append(midpoint)
                    sources_seen.add(posting.source)

    # Find the most represented currency
    if not salary_values_by_currency:
        return SalaryData(
            role=role,
            location=country or "Global",
            min_salary=None,
            max_salary=None,
            median_salary=None,
            currency="USD",
            sample_count=0,
            sources=list(sources_seen),
        )

    # Use the currency with the most data points
    best_currency = max(
        salary_values_by_currency,
        key=lambda c: len(salary_values_by_currency[c]),
    )
    values = sorted(salary_values_by_currency[best_currency])

    median_idx = len(values) // 2
    median = values[median_idx] if values else None

    return SalaryData(
        role=role,
        location=country or "Global",
        min_salary=min(values) if values else None,
        max_salary=max(values) if values else None,
        median_salary=median,
        currency=best_currency,
        sample_count=len(values),
        sources=list(sources_seen),
    )


@tool
def parse_salaries(job_search_result: dict, country: Optional[str] = None) -> dict:
    """
    Extract and aggregate salary information from job postings.

    IMPORTANT: Only returns salary data that exists in the actual postings.
    If fewer than 5 postings have salary data, the result is flagged
    as having insufficient evidence.

    Args:
        job_search_result: Serialized JobSearchResult from search_jobs tool.
        country: Optional ISO country code to filter by (e.g., 'IN', 'US').

    Returns:
        Dictionary with salary statistics and sample count.
    """
    result = JobSearchResult.model_validate(job_search_result)
    role = result.search_params.get("role", "Unknown Role")

    salary_data = compute_salary_statistics(
        postings=result.postings,
        role=role,
        country=country,
    )

    # Determine reliability
    is_reliable = salary_data.sample_count >= MIN_RELIABLE_SAMPLE
    evidence_note = (
        f"Based on {salary_data.sample_count} postings with disclosed salary "
        f"out of {len(result.postings)} total analyzed."
    )

    if not is_reliable:
        reliability_warning = (
            "⚠️ Insufficient evidence: fewer than 5 postings disclosed salary. "
            "These figures should not be treated as market benchmarks."
        )
    else:
        reliability_warning = None

    output = salary_data.model_dump()
    output["evidence_note"] = evidence_note
    output["is_reliable"] = is_reliable
    if reliability_warning:
        output["reliability_warning"] = reliability_warning

    logger.info(
        "Salary parsing complete",
        extra={
            "role": role,
            "sample_count": salary_data.sample_count,
            "is_reliable": is_reliable,
        },
    )

    return output
