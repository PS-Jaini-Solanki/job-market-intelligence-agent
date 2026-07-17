"""
models/resume.py
================
Pydantic models for resume data and gap analysis.

The Resume model stores what the user actually has.
The ResumeAnalysis model compares it to what the market wants.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from models.skill import Skill, SkillGap

        
class Resume(BaseModel):
    """
    Represents a parsed user resume.

    raw_text is the full extracted text from the file.
    extracted_skills are found by the skill extraction tool.
    """

    id: UUID = Field(default_factory=uuid4)
    file_name: str = Field(description="Original filename of the uploaded resume")
    raw_text: str = Field(description="Full text extracted from the resume file")
    extracted_skills: list[Skill] = Field(
        default_factory=list,
        description="Skills identified in the resume by the skill extractor",
    )
    experience_years: Optional[float] = Field(
        default=None,
        description="Total years of experience extracted from resume, if determinable",
    )
    education: list[str] = Field(
        default_factory=list,
        description="Education entries extracted from the resume",
    )
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    def get_skill_names(self) -> list[str]:
        """
        Get a simple list of normalized skill names from this resume.

        Returns:
            List of normalized skill name strings for easy comparison.
        """
        return [s.normalized_name for s in self.extracted_skills]

    def has_skill(self, skill_name: str) -> bool:
        """
        Check if the resume contains a particular skill.

        Comparison is case-insensitive and uses normalized names.

        Args:
            skill_name: The skill to look for.

        Returns:
            True if the skill is found, False otherwise.
        """
        normalized_query = skill_name.lower().strip()
        return any(
            s.normalized_name.lower() == normalized_query
            for s in self.extracted_skills
        )


class ResumeAnalysis(BaseModel):
    """
    The result of comparing a resume against current market requirements.

    HALLUCINATION PREVENTION:
    - matching_skills: Skills present in BOTH the resume AND market data
    - missing_skills: Skills in market data but NOT in the resume
    - gap_score: A computed metric (0–100), where 100 = perfect market fit
    All findings are derived from actual data, never generated.
    """

    resume_id: UUID
    query_context: str = Field(
        description="The job role/market context this analysis is for. "
                    "E.g. 'AI Engineer in India'"
    )

    # Skills the user HAS that the market wants
    matching_skills: list[Skill] = Field(
        default_factory=list,
        description="Skills found in both the resume and market requirements",
    )
    # Skills the market wants that the user DOES NOT have
    missing_skills: list[SkillGap] = Field(
        default_factory=list,
        description="High-demand skills missing from the resume, ranked by priority",
    )
    # Skills the user has that are rare in job postings
    bonus_skills: list[Skill] = Field(
        default_factory=list,
        description="Skills the user has that appear in < 5% of job postings "
                    "(niche differentiators)",
    )

    gap_score: float = Field(
        description="Market fit score from 0 to 100. "
                    "Computed as: (matching_count / top_skills_count) × 100"
    )

    total_postings_analyzed: int = Field(
        description="Number of job postings used for this analysis"
    )
    sources: list[str] = Field(
        description="Data sources used for market data"
    )
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    def get_high_priority_gaps(self) -> list[SkillGap]:
        """
        Return only HIGH priority skill gaps.

        Returns:
            SkillGap items where priority == 'HIGH'.
        """
        return [gap for gap in self.missing_skills if gap.priority == "HIGH"]

    def summary(self) -> str:
        """
        Generate a concise summary of the analysis for display.

        Returns:
            Multi-line summary string.
        """
        high_gaps = self.get_high_priority_gaps()
        matching_names = [s.normalized_name for s in self.matching_skills]
        gap_names = [g.skill.normalized_name for g in high_gaps]

        return (
            f"Market Fit Score: {self.gap_score:.1f}/100\n"
            f"Matching Skills ({len(self.matching_skills)}): "
            f"{', '.join(matching_names[:5]) or 'None'}\n"
            f"High-Priority Gaps ({len(high_gaps)}): "
            f"{', '.join(gap_names[:5]) or 'None'}\n"
            f"Based on: {self.total_postings_analyzed} job postings "
            f"from {', '.join(self.sources)}"
        )
