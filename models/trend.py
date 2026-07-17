"""
models/trend.py
===============
Pydantic models for technology trend data.

Trend data is collected from two sources:
1. GitHub (stars, forks, contributors, release frequency)
2. Job postings (how often a tech appears in job descriptions)

Combining both gives a more reliable signal than either alone.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class GitHubStats(BaseModel):
    """
    Raw statistics pulled from the GitHub API for a repository or topic.

    All values are factual — fetched directly from the API response.
    """

    repository: Optional[str] = Field(
        default=None,
        description="GitHub repository name, e.g. 'langchain-ai/langchain'",
    )
    stars: int = Field(default=0, description="Total star count")
    forks: int = Field(default=0, description="Total fork count")
    open_issues: int = Field(default=0, description="Number of open issues")
    contributors: Optional[int] = Field(
        default=None,
        description="Number of contributors (requires extra API call)",
    )
    recent_commits_30d: Optional[int] = Field(
        default=None,
        description="Commits in the last 30 days (activity signal)",
    )
    last_release: Optional[datetime] = Field(
        default=None,
        description="Date of the most recent release",
    )
    language: Optional[str] = Field(
        default=None,
        description="Primary programming language",
    )
    license: Optional[str] = Field(
        default=None,
        description="License type, e.g. 'MIT', 'Apache-2.0'",
    )
    source_url: str = Field(
        description="GitHub URL for this repository"
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)


class TechTrend(BaseModel):
    """
    Combined trend signal for a single technology.

    Aggregates data from GitHub and job postings to produce
    a unified view of how popular/trending a technology is.

    HALLUCINATION PREVENTION:
    - github_stats comes from real GitHub API calls
    - job_count comes from real job search results
    - popularity_score is a computed formula, not an LLM estimate
    - All sources are listed explicitly
    """

    technology: str = Field(
        description="Technology name in its normalized form. E.g. 'LangGraph'"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of what this technology is",
    )

    # GitHub data
    github_stats: Optional[GitHubStats] = Field(
        default=None,
        description="GitHub repository statistics for this technology",
    )

    # Job market data
    job_count: int = Field(
        default=0,
        description="Number of job postings mentioning this technology",
    )
    total_postings_analyzed: int = Field(
        default=0,
        description="Total job postings in the dataset (for computing percentages)",
    )

    # Growth signal (requires at least 2 snapshots to compute)
    growth_rate_30d: Optional[float] = Field(
        default=None,
        description="% change in job mentions over the last 30 days. "
                    "None if insufficient historical data.",
    )

    sources: list[str] = Field(
        default_factory=list,
        description="All data sources contributing to this trend signal",
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def job_demand_percentage(self) -> float:
        """
        What percentage of analyzed job postings mention this technology.

        Returns 0.0 if no postings were analyzed.
        """
        if self.total_postings_analyzed == 0:
            return 0.0
        return round((self.job_count / self.total_postings_analyzed) * 100, 1)

    @computed_field
    @property
    def popularity_score(self) -> float:
        """
        Composite popularity score (0–100) combining GitHub and job data.

        Formula:
          github_component (0–50): log-scaled GitHub stars
          job_component (0–50): job demand percentage (capped at 50%)

        This is a heuristic — we document it so users understand
        exactly how the score is computed.
        """
        import math

        github_component = 0.0
        if self.github_stats and self.github_stats.stars > 0:
            # Log scale: 100k stars → 50 points, 1k stars → 25 points
            github_component = min(50.0, (math.log10(self.github_stats.stars) / 5) * 50)

        # Cap at 50 points from job demand
        job_component = min(50.0, self.job_demand_percentage)

        return round(github_component + job_component, 2)

    def as_evidence_string(self) -> str:
        """
        Return a factual evidence string for inclusion in reports.

        Example:
            "LangGraph: 8,450 GitHub stars | 12/100 job postings (12.0%)"
        """
        stars = (
            f"{self.github_stats.stars:,} GitHub stars"
            if self.github_stats
            else "No GitHub data"
        )
        return (
            f"{self.technology}: {stars} | "
            f"{self.job_count}/{self.total_postings_analyzed} job postings "
            f"({self.job_demand_percentage}%)"
        )


class TrendSnapshot(BaseModel):
    """
    A point-in-time collection of TechTrend objects.

    Taking snapshots over time lets us compute growth rates.
    """

    trends: list[TechTrend] = Field(
        description="All technology trends in this snapshot"
    )
    query_context: str = Field(
        description="What this snapshot was collected for. "
                    "E.g. 'AI Engineer tools, India market'"
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    def get_ranked(self, top_n: int = 20) -> list[TechTrend]:
        """
        Return trends ranked by popularity_score (highest first).

        Args:
            top_n: Maximum number of trends to return.

        Returns:
            Top N trends by popularity score.
        """
        ranked = sorted(
            self.trends,
            key=lambda t: t.popularity_score,
            reverse=True,
        )
        return ranked[:top_n]

    def get_by_technology(self, name: str) -> Optional[TechTrend]:
        """
        Find a specific technology in this snapshot.

        Args:
            name: Technology name (case-insensitive).

        Returns:
            The matching TechTrend, or None if not found.
        """
        name_lower = name.lower()
        for trend in self.trends:
            if trend.technology.lower() == name_lower:
                return trend
        return None
