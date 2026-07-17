"""
models/agent.py
===============
Pydantic models for inter-agent communication.

In LangGraph, agents communicate by reading and writing to a shared
"state" object. All messages between agents use these models, which
ensures type safety and prevents silent data corruption.

Key rule: No raw dicts between agents — always use these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """
    All possible task types the planner can assign to agents.

    Each enum value maps to a specific agent in the graph:
    - COLLECT_JOBS → MarketCollectionAgent
    - EXTRACT_SKILLS → SkillExtractionAgent
    - ANALYZE_TRENDS → TrendAnalysisAgent
    - ANALYZE_RESUME → ResumeAnalysisAgent
    - GENERATE_RECOMMENDATIONS → RecommendationAgent
    - GENERATE_REPORT → ReportGenerationAgent
    """

    COLLECT_JOBS = "collect_jobs"
    ANALYZE_SALARIES = "analyze_salaries"
    EXTRACT_SKILLS = "extract_skills"
    ANALYZE_TRENDS = "analyze_trends"
    ANALYZE_RESUME = "analyze_resume"
    COMPARE_COUNTRIES = "compare_countries"
    GENERATE_RECOMMENDATIONS = "generate_recommendations"
    GENERATE_REPORT = "generate_report"


class TaskStatus(str, Enum):
    """Lifecycle states for an agent task."""

    PENDING = "pending"      # Not yet started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Encountered an error
    SKIPPED = "skipped"      # Skipped by planner (not needed for this query)


class MCPServerType(str, Enum):
    """The MCP servers available to agents."""

    GITHUB = "github"
    FILESYSTEM = "filesystem"
    SEARCH = "search"
    REDDIT = "reddit"


class AgentTask(BaseModel):
    """
    A single unit of work assigned by the Planner to a specific agent.

    The planner breaks every user request into a list of AgentTasks.
    Each task is independent and can be executed by its assigned agent.
    """

    task_id: UUID = Field(default_factory=uuid4)
    task_type: TaskType = Field(description="What kind of work this task requires")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for this task. "
                    "Example: {'country': 'IN', 'role': 'AI Engineer'}",
    )
    required_mcp_servers: list[MCPServerType] = Field(
        default_factory=list,
        description="Which MCP servers this task needs access to",
    )
    depends_on: list[UUID] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task starts",
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Optional[Any] = Field(
        default=None,
        description="The output of this task once completed",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if this task failed",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def duration_seconds(self) -> Optional[float]:
        """
        Calculate how long this task took to execute.

        Returns:
            Duration in seconds, or None if task hasn't completed.
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class PlannerOutput(BaseModel):
    """
    The complete plan produced by the Planner agent.

    The planner analyzes the user's request and produces an ordered
    list of tasks for other agents to execute. The planner also
    explains its reasoning so users can see how decisions are made.
    """

    tasks: list[AgentTask] = Field(
        description="Ordered list of tasks to execute"
    )
    reasoning: str = Field(
        description="Explanation of why these tasks were chosen"
    )
    query_type: str = Field(
        description="Detected type of user query. "
                    "Examples: 'job_market', 'salary', 'resume_analysis', "
                    "'tech_trend', 'country_comparison'"
    )
    requires_resume: bool = Field(
        default=False,
        description="True if this query requires an uploaded resume",
    )
    estimated_duration_seconds: Optional[int] = Field(
        default=None,
        description="Rough estimate of how long execution will take",
    )

    def get_tasks_by_type(self, task_type: TaskType) -> list[AgentTask]:
        """Get all tasks of a specific type."""
        return [t for t in self.tasks if t.task_type == task_type]

    def get_mcp_servers_needed(self) -> list[MCPServerType]:
        """Get the complete set of MCP servers needed across all tasks."""
        servers: set[MCPServerType] = set()
        for task in self.tasks:
            servers.update(task.required_mcp_servers)
        return list(servers)


class AgentMessage(BaseModel):
    """
    A message passed between agents during execution.

    All inter-agent communication goes through these typed messages
    to maintain a clear audit trail of what happened.
    """

    message_id: UUID = Field(default_factory=uuid4)
    from_agent: str = Field(description="Name of the sending agent")
    to_agent: str = Field(description="Name of the receiving agent")
    content: Any = Field(description="The message payload (any serializable type)")
    message_type: str = Field(
        description="Type of message: 'result', 'error', 'status_update'"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReportSection(BaseModel):
    """A single section in the final market intelligence report."""

    title: str
    content: str = Field(description="Markdown-formatted section content")
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence strings backing the claims in this section",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Data sources for this section",
    )


class MarketReport(BaseModel):
    """
    The final market intelligence report.

    This is what the user receives at the end of agent execution.

    HALLUCINATION PREVENTION:
    Every claim in findings must have a corresponding entry in evidence.
    The report should not contain any statistic without a source.
    """

    query: str = Field(description="The original user query")
    executive_summary: str = Field(
        description="A 3–5 sentence overview of key findings"
    )
    sections: list[ReportSection] = Field(
        description="Detailed report sections"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations based on the analysis",
    )
    data_limitations: list[str] = Field(
        default_factory=list,
        description="Honest notes about what the data cannot tell us",
    )
    total_postings_analyzed: int = Field(
        default=0,
        description="Total job postings analyzed for this report",
    )
    sources: list[str] = Field(
        description="All data sources used"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_seconds: Optional[float] = None

    def to_markdown(self) -> str:
        """
        Render the full report as a Markdown document.

        Returns:
            A formatted Markdown string ready for display in Chainlit.
        """
        lines = [
            f"# Job Market Intelligence Report",
            f"**Query:** {self.query}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Data:** {self.total_postings_analyzed} job postings analyzed",
            f"**Sources:** {', '.join(self.sources)}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            self.executive_summary,
            "",
        ]

        for section in self.sections:
            lines.extend([
                f"## {section.title}",
                "",
                section.content,
                "",
            ])
            if section.evidence:
                lines.append("**Evidence:**")
                for ev in section.evidence:
                    lines.append(f"- {ev}")
                lines.append("")

        if self.recommendations:
            lines.extend(["## Recommendations", ""])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        if self.data_limitations:
            lines.extend(["## Data Limitations", ""])
            for lim in self.data_limitations:
                lines.append(f"- ⚠️ {lim}")
            lines.append("")

        return "\n".join(lines)
