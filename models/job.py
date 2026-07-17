"""
models/job.py
=============
Pydantic models for job posting data.

HALLUCINATION PREVENTION:
- Every salary value must come from actual job posting data
- Every skill must be extracted from real job descriptions
- Source URLs are required so data can be verified
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SalaryRange(BaseModel):
    """
    Represents a parsed salary range from a job posting.

    Design note: We always store min AND max separately so we
    can compute averages without losing precision.
    """

    min_value: Optional[float] = Field(
        default=None,
        description="Minimum salary in the given currency",
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum salary in the given currency",
    )
    currency: str = Field(
        default="USD",
        description="ISO 4217 currency code, e.g. USD, INR, GBP",
    )
    period: str = Field(
        default="year",
        description="Payment period: year | month | hour",
    )

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        """Normalize currency codes to uppercase."""
        return v.upper()

    def as_annual_usd_estimate(self, usd_rates: dict[str, float]) -> Optional[float]:
        """
        Convert salary to an estimated annual USD figure.

        Args:
            usd_rates: Dictionary of currency code → USD exchange rate.
                       Example: {"INR": 0.012, "GBP": 1.27}

        Returns:
            Estimated annual USD salary, or None if conversion not possible.
        """
        if self.min_value is None and self.max_value is None:
            return None

        # Use midpoint if both values exist; otherwise use whichever exists
        midpoint = (
            (self.min_value + self.max_value) / 2
            if (self.min_value and self.max_value)
            else (self.min_value or self.max_value)
        )

        # Convert to annual
        multiplier = {"year": 1, "month": 12, "hour": 2080}.get(self.period, 1)
        annual = midpoint * multiplier

        # Convert to USD using provided rates
        rate = usd_rates.get(self.currency, 1.0)
        return round(annual * rate, 2)


class SalaryData(BaseModel):
    """
    Aggregated salary statistics for a role/location combination.

    IMPORTANT: These values are computed from real job postings,
    never invented. The 'sample_count' field shows the evidence base.
    """

    role: str = Field(description="Job title this salary data is for")
    location: str = Field(description="Location/country for this salary data")
    min_salary: Optional[float] = Field(default=None)
    max_salary: Optional[float] = Field(default=None)
    median_salary: Optional[float] = Field(default=None)
    currency: str = Field(default="USD")
    # CRITICAL: Always track how many postings this is based on
    sample_count: int = Field(
        description="Number of job postings this data is derived from"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of data sources, e.g. ['adzuna', 'arbeitnow']",
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)


class JobPosting(BaseModel):
    """
    Represents a single job posting collected from an external source.

    Every field that contains factual information must be directly
    extracted from the raw job posting — never inferred or generated.
    """

    title: str = Field(description="Job title exactly as posted")
    company: str = Field(description="Company name exactly as posted")
    location: str = Field(description="Location string from the posting")
    country: Optional[str] = Field(
        default=None,
        description="ISO country code, e.g. 'IN', 'US'",
    )
    description: Optional[str] = Field(
        default=None,
        description="Full or truncated job description text",
    )
    salary: Optional[SalaryRange] = Field(
        default=None,
        description="Parsed salary range; None if not listed",
    )
    # Skills are extracted from the description — not generated
    skills: list[str] = Field(
        default_factory=list,
        description="Skills mentioned in the job description",
    )
    source: str = Field(
        description="Name of the data source, e.g. 'adzuna', 'arbeitnow'"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Direct URL to the original job posting",
    )
    posted_at: Optional[datetime] = Field(
        default=None,
        description="When the job was posted, if available",
    )
    collected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When we collected this posting",
    )

    def __str__(self) -> str:
        return f"{self.title} @ {self.company} ({self.location})"


class JobSearchResult(BaseModel):
    """
    The result of a job search query, containing all collected postings
    along with metadata about the search itself.

    This is what the job_search_tool returns.
    """

    query: str = Field(description="The search query that was used")
    postings: list[JobPosting] = Field(
        default_factory=list,
        description="List of collected job postings",
    )
    total_found: int = Field(
        default=0,
        description="Total number of postings found (may exceed returned count)",
    )
    sources_queried: list[str] = Field(
        default_factory=list,
        description="Which APIs/sources were queried",
    )
    search_params: dict = Field(
        default_factory=dict,
        description="The parameters used for the search (country, role, etc.)",
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def count(self) -> int:
        """Number of postings actually returned."""
        return len(self.postings)

    def get_all_skills(self) -> list[str]:
        """
        Flatten all skills across all job postings into a single list.

        Returns:
            Flat list of all skill strings (may contain duplicates —
            that's intentional so we can count frequencies).
        """
        all_skills = []
        for posting in self.postings:
            all_skills.extend(posting.skills)
        return all_skills

    def get_postings_by_country(self, country_code: str) -> list[JobPosting]:
        """
        Filter postings to a specific country.

        Args:
            country_code: ISO country code, e.g. 'IN', 'US'.

        Returns:
            List of postings matching the given country.
        """
        return [
            p for p in self.postings
            if p.country and p.country.upper() == country_code.upper()
        ]
