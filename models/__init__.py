"""
models/__init__.py
==================
Re-exports all Pydantic models for convenient importing.

Usage:
    from models import JobPosting, Skill, Resume, AgentTask
"""

from models.job import JobPosting, JobSearchResult, SalaryData, SalaryRange
from models.skill import Skill, SkillCategory, SkillStatistic
from models.resume import Resume, ResumeAnalysis
from models.agent import (
    AgentTask,
    TaskType,
    TaskStatus,
    PlannerOutput,
    AgentMessage,
    MarketReport,
    ReportSection,
)
from models.trend import TechTrend, TrendSnapshot

__all__ = [
    # Job models
    "JobPosting",
    "JobSearchResult",
    "SalaryData",
    "SalaryRange",
    # Skill models
    "Skill",
    "SkillCategory",
    "SkillStatistic",
    # Resume models
    "Resume",
    "ResumeAnalysis",
    # Agent models
    "AgentTask",
    "TaskType",
    "TaskStatus",
    "PlannerOutput",
    "AgentMessage",
    "MarketReport",
    "ReportSection",
    # Trend models
    "TechTrend",
    "TrendSnapshot",
]
