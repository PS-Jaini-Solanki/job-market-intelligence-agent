"""
agents/report_generation.py
============================
Report Generation Agent — compiles all findings into the final report.

RESPONSIBILITY:
- Read all data from state (skills, salary, trends, resume analysis)
- Use ReportBuilder to construct a structured MarketReport
- Ask the LLM to write an executive summary (LLM only sees real data)
- Store final report in state

This is the LAST agent to run in the graph.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.agent import MCPCallRecord
from models.trend import TrendSnapshot
from models.resume import ResumeAnalysis
from models.skill import Skill, SkillStatistic
from prompts.report_prompt import REPORT_SYSTEM_PROMPT, build_report_prompt
from services.llm_service import get_llm_service
from services.report_service import ReportBuilder

logger = logging.getLogger(__name__)


async def run_report_generation(state: GraphState) -> dict:
    """
    Report Generation agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with final_report.
    """
    logger.info("Report generation starting")

    # If the user asked a resume question but no resume was ever attached,
    # there's nothing real to report on. market_collection/skill_extraction
    # still ran (they always do, so data is ready in case a resume shows
    # up), but presenting that side-collected, unrelated market data as a
    # polished "Job Market Intelligence Report" — right next to a chat
    # answer that correctly says no resume was found — is a direct,
    # visible contradiction: the report looks like a complete, successful
    # analysis for a question that was never actually answered. Skip the
    # report entirely rather than manufacture one.
    if state.get("query_intent") == "resume_analysis" and not state.get("resume_data"):
        logger.info("Skipping report generation — no resume was attached to analyze")
        return {
            "final_report": None,
            "completed_agents": state.get("completed_agents", []) + ["report_generation"],
            "messages": [
                AIMessage(content="📄 No report generated — no resume was attached to analyze.")
            ],
        }

    query = state["user_query"]
    role = state.get("role", "Software Engineer")
    country = state.get("country_filter")
    skill_stats_raw = state.get("skill_statistics") or []
    salary_raw = state.get("salary_data")
    trend_raw = state.get("trend_data")
    resume_analysis_raw = state.get("resume_analysis")
    comparison_raw = state.get("country_comparison")
    recommendations = state.get("recommendations") or []

    # Initialize the report builder
    builder = ReportBuilder(query=query)

    # --- Add Skills Section ---
    if skill_stats_raw:
        total_postings = skill_stats_raw[0]["total_postings"] if skill_stats_raw else 0

        # Reconstruct SkillStatistic objects for the builder
        # (builder expects model objects, not raw dicts)
        from models.skill import SkillCategory
        stats_for_builder = []
        for s in skill_stats_raw[:20]:
            skill = Skill(
                raw_name=s["skill_name"],
                normalized_name=s["skill_name"],
                category=SkillCategory(s.get("category", "other")),
            )
            stat = SkillStatistic(
                skill=skill,
                occurrence_count=s["occurrence_count"],
                total_postings_analyzed=s["total_postings"],
                sources=s.get("sources", ["adzuna", "arbeitnow"]),
                country=country,
            )
            stats_for_builder.append(stat)

        builder.add_skills_section(
            skill_stats=stats_for_builder,
            total_postings=total_postings,
            country=country,
        )

    # --- Add Salary Section ---
    if salary_raw:
        builder.add_salary_section(
            role=role,
            min_sal=salary_raw.get("min_salary"),
            max_sal=salary_raw.get("max_salary"),
            median_sal=salary_raw.get("median_salary"),
            currency=salary_raw.get("currency", "USD"),
            sample_count=salary_raw.get("sample_count", 0),
            sources=salary_raw.get("sources", []),
            country=country,
        )

    # --- Add Trends Section ---
    if trend_raw:
        try:
            snapshot = TrendSnapshot.model_validate(trend_raw)
            builder.add_trends_section(snapshot, top_n=10)
        except Exception as e:
            logger.warning("Could not build trends section", extra={"error": str(e)})

    # --- Add Resume Section ---
    if resume_analysis_raw:
        try:
            analysis = ResumeAnalysis.model_validate(resume_analysis_raw)
            builder.add_resume_section(analysis)
        except Exception as e:
            logger.warning("Could not build resume section", extra={"error": str(e)})

    # --- Add Country Comparison Section ---
    if comparison_raw and "error" not in comparison_raw:
        try:
            builder.add_country_comparison_section(comparison_raw)
        except Exception as e:
            logger.warning("Could not build comparison section", extra={"error": str(e)})

    # --- Add Recommendations ---
    for rec in recommendations:
        builder.add_recommendation(rec)

    # --- Generate Executive Summary using LLM ---
    summary_data = {
        "query": query,
        "role": role,
        "country": country or "Global",
        "top_skills": [s["skill_name"] for s in skill_stats_raw[:10]],
        "total_postings": skill_stats_raw[0]["total_postings"] if skill_stats_raw else 0,
        "sources": list({s for stat in skill_stats_raw for s in stat.get("sources", [])}),
        "has_salary_data": bool(salary_raw and salary_raw.get("sample_count", 0) > 0),
        "has_trend_data": bool(trend_raw),
        "has_resume_analysis": bool(resume_analysis_raw),
        "has_country_comparison": bool(comparison_raw and "error" not in comparison_raw),
    }

    executive_summary = ""
    try:
        llm = get_llm_service()
        executive_summary = await llm.generate(
            user_prompt=build_report_prompt(summary_data),
            system_prompt=REPORT_SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("LLM summary generation failed", extra={"error": str(e)})
        executive_summary = (
            f"Analysis of {summary_data['total_postings']} job postings "
            f"for {role} in {country or 'global market'}. "
            f"Top skills identified: {', '.join(summary_data['top_skills'][:5])}."
        )

    # --- Build final report ---
    report = builder.build(executive_summary=executive_summary.strip())

    # --- Persist report via filesystem MCP (visible in UI) ---
    mcp_calls = state.get("mcp_calls", []) or []
    save_result = None
    try:
        from mcp_servers.registry import get_registry
        registry = get_registry()
        query_id = state.get("query_id") or "report"
        save_result = await registry.call_tool(
            "filesystem",
            "save_report",
            {
                "filename": f"market_report_{query_id}.md",
                "content": report.to_markdown(),
            },
        )
        mcp_calls = [
            *mcp_calls,
            MCPCallRecord(
                agent="report_generation",
                server="filesystem",
                tool="save_report",
                arguments={"filename": f"market_report_{query_id}.md"},
                success=save_result.success,
                summary=(
                    f"Saved to {save_result.result.get('saved_to', 'reports/')}"
                    if save_result.success and save_result.result
                    else (save_result.error or "save failed")
                ),
            ).model_dump(),
        ]
    except Exception as e:
        logger.warning("Could not save report via filesystem MCP", extra={"error": str(e)})

    logger.info(
        "Report generation complete",
        extra={
            "sections": len(report.sections),
            "recommendations": len(report.recommendations),
            "execution_time": report.execution_time_seconds,
        },
    )

    return {
        "final_report": report.model_dump(),
        "executive_summary": executive_summary,
        "mcp_calls": mcp_calls,
        "completed_agents": state.get("completed_agents", []) + ["report_generation"],
        "messages": [
            AIMessage(
                content=(
                    f"📊 **Report Complete!**\n\n"
                    f"{executive_summary[:300]}{'...' if len(executive_summary) > 300 else ''}"
                )
            )
        ],
    }
