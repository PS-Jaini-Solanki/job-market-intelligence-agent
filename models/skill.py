"""
models/skill.py
===============
Pydantic models for technology skills.

Key design decisions:
1. Every skill has a "normalized_name" — this prevents double-counting
   variations like "AWS", "Amazon Web Services", "Amazon AWS".
2. SkillStatistic always tracks how many total postings were analyzed,
   so percentages are honest (e.g., "184 of 220 postings", not "84%").
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class SkillCategory(str, Enum):
    """
    Broad categories for organizing skills.

    Using an Enum prevents typos and makes filtering easy.
    """

    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    DATABASE = "database"
    CLOUD = "cloud"
    DEVOPS = "devops"
    AI_ML = "ai_ml"
    DATA_ENGINEERING = "data_engineering"
    SOFT_SKILL = "soft_skill"
    PROTOCOL = "protocol"
    OTHER = "other"


class Skill(BaseModel):
    """
    Represents a single technology skill.

    The 'raw_name' is what was found in the job description.
    The 'normalized_name' is the canonical form after normalization.
    """

    raw_name: str = Field(
        description="Skill name exactly as it appeared in the source text"
    )
    normalized_name: str = Field(
        description="Canonical skill name after normalization. "
                    "Example: 'Amazon Web Services' → 'AWS'"
    )
    category: SkillCategory = Field(
        default=SkillCategory.OTHER,
        description="Broad category this skill belongs to",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names this skill might appear as",
    )

    def __hash__(self) -> int:
        """Make Skill hashable so it can be used in sets."""
        return hash(self.normalized_name.lower())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Skill):
            return self.normalized_name.lower() == other.normalized_name.lower()
        return False


class SkillStatistic(BaseModel):
    """
    Aggregated statistics for a single skill across job postings.

    HALLUCINATION PREVENTION:
    We always report "X out of Y postings" rather than a bare percentage.
    The 'sources' field tells users exactly where the data comes from.
    """

    skill: Skill = Field(description="The skill this statistic is about")
    occurrence_count: int = Field(
        description="Number of job postings that mention this skill"
    )
    total_postings_analyzed: int = Field(
        description="Total number of job postings in the dataset"
    )
    sources: list[str] = Field(
        description="Data sources used, e.g. ['adzuna', 'arbeitnow']"
    )
    country: Optional[str] = Field(
        default=None,
        description="Country filter applied, or None for global",
    )
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def percentage(self) -> float:
        """
        Percentage of analyzed postings that mention this skill.

        Returns 0.0 if total_postings_analyzed is 0 to avoid division by zero.
        """
        if self.total_postings_analyzed == 0:
            return 0.0
        return round(
            (self.occurrence_count / self.total_postings_analyzed) * 100, 1
        )

    def as_evidence_string(self) -> str:
        """
        Return a factual evidence string suitable for reports.

        Example output:
            "Python: 184 / 220 postings (83.6%) — Source: adzuna, arbeitnow"

        This format prevents hallucinated statistics — every claim
        is backed by a count and a source.
        """
        source_str = ", ".join(self.sources)
        return (
            f"{self.skill.normalized_name}: "
            f"{self.occurrence_count} / {self.total_postings_analyzed} postings "
            f"({self.percentage}%) — Source: {source_str}"
        )


class SkillGap(BaseModel):
    """
    Represents a gap between a user's skills and market requirements.
    Used in resume analysis to show what skills to prioritize learning.
    """

    skill: Skill
    market_demand_percentage: float = Field(
        description="What percentage of analyzed job postings require this skill"
    )
    priority: str = Field(
        description="Priority level: HIGH | MEDIUM | LOW based on demand"
    )
    learning_resources: list[str] = Field(
        default_factory=list,
        description="Suggested learning resources (sourced, not invented)",
    )

    @classmethod
    def from_statistic(cls, stat: SkillStatistic) -> "SkillGap":
        """
        Create a SkillGap from a SkillStatistic.

        Priority is determined by market demand percentage:
        - HIGH: > 30% of job postings require it
        - MEDIUM: 10–30%
        - LOW: < 10%
        """
        pct = stat.percentage
        priority = "HIGH" if pct > 30 else ("MEDIUM" if pct > 10 else "LOW")
        return cls(
            skill=stat.skill,
            market_demand_percentage=pct,
            priority=priority,
        )
