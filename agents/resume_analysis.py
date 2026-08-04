"""
agents/resume_analysis.py
==========================
Resume Analysis Agent — compares user's resume to market requirements.

RESPONSIBILITY:
- Parse the uploaded resume file (PDF/DOCX/TXT)
- Extract skills from the resume
- Compare with skills from job postings (from state)
- Compute a market fit score and gap analysis

This agent only runs when a resume has been uploaded.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.agent import MCPCallRecord
from models.resume import Resume, ResumeAnalysis
from models.skill import Skill, SkillGap, SkillStatistic
from tools.resume_parser_tool import extract_text_from_file, estimate_experience_years
from tools.skill_extractor_tool import extract_skills_from_text, _infer_skill_category

logger = logging.getLogger(__name__)

# Top N market skills to compare against
TOP_MARKET_SKILLS = 30


async def run_resume_analysis(state: GraphState) -> dict:
    """
    Resume Analysis agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with resume_data and resume_analysis.
    """
    resume_path = state.get("resume_path")

    if not resume_path:
        logger.info("No resume path in state — skipping resume analysis")
        return {
            "completed_agents": state.get("completed_agents", []) + ["resume_analysis"],
            "messages": [AIMessage(content="📄 Resume analysis skipped — no resume uploaded.")],
        }

    logger.info("Resume analysis starting", extra={"path": resume_path})

    # Step 1: Read the resume through the filesystem MCP (visible in UI),
    # then parse the content with the resume_parser_tool.
    mcp_calls = state.get("mcp_calls", []) or []
    from mcp_servers.registry import get_registry
    registry = get_registry()
    read_result = await registry.call_tool(
        "filesystem", "read_file", {"file_path": resume_path}
    )
    mcp_calls = [
        *mcp_calls,
        MCPCallRecord(
            agent="resume_analysis",
            server="filesystem",
            tool="read_file",
            arguments={"file_path": resume_path},
            success=read_result.success,
            summary=(
                "Located resume file for parsing"
                if read_result.success else (read_result.error or "read failed")
            ),
        ).model_dump(),
    ]

    raw_text = ""
    try:
        if read_result.success and read_result.result and "content" in read_result.result:
            raw_text = read_result.result["content"]
        else:
            # Binary files (PDF/DOCX) need the parser — read them directly.
            raw_text = extract_text_from_file(resume_path)
    except Exception as e:
        logger.error("Resume text extraction failed", extra={"error": str(e)})
        raw_text = ""

    skill_names = extract_skills_from_text(raw_text)
    experience_years = estimate_experience_years(raw_text)

    resume_skills = [
        Skill(
            raw_name=name,
            normalized_name=name,
            category=_infer_skill_category(name),
        )
        for name in skill_names
    ]

    import os
    resume = Resume(
        file_name=os.path.basename(resume_path),
        raw_text=raw_text,
        extracted_skills=resume_skills,
        experience_years=experience_years,
    )

    # Step 2: Get market skill statistics from state
    skill_stats_raw = state.get("skill_statistics") or []
    if not skill_stats_raw:
        logger.warning("No skill statistics available for resume gap analysis")
        return {
            "resume_data": resume.model_dump(),
            "mcp_calls": mcp_calls,
            "errors": state.get("errors", []) + [
                "Resume parsed but gap analysis skipped — no market data available"
            ],
            "completed_agents": state.get("completed_agents", []) + ["resume_analysis"],
            "messages": [
                AIMessage(
                    content=f"📄 Resume parsed — found {len(resume_skills)} skills. "
                    "Gap analysis unavailable (run job collection first)."
                )
            ],
        }

    # Step 3: Compute gap analysis
    total_postings = skill_stats_raw[0]["total_postings"] if skill_stats_raw else 0
    top_market_stats = skill_stats_raw[:TOP_MARKET_SKILLS]

    user_skill_names = {s.normalized_name.lower() for s in resume_skills}

    matching_skills: list[Skill] = []
    missing_gaps: list[SkillGap] = []
    bonus_skills: list[Skill] = []

    # Skills in the market
    market_skill_names_lower = {s["skill_name"].lower() for s in top_market_stats}

    for stat in top_market_stats:
        skill_name = stat["skill_name"]
        pct = stat["percentage"]

        skill_obj = Skill(
            raw_name=skill_name,
            normalized_name=skill_name,
            category=_infer_skill_category(skill_name),
        )

        if skill_name.lower() in user_skill_names:
            matching_skills.append(skill_obj)
        else:
            # This is a market skill the user doesn't have
            priority = "HIGH" if pct > 30 else ("MEDIUM" if pct > 10 else "LOW")
            missing_gaps.append(SkillGap(
                skill=skill_obj,
                market_demand_percentage=pct,
                priority=priority,
            ))

    # Find bonus skills (user has them but they're not in top market skills)
    for user_skill in resume_skills:
        if user_skill.normalized_name.lower() not in market_skill_names_lower:
            bonus_skills.append(user_skill)

    # Compute gap score: what % of top market skills does the user have?
    gap_score = (
        (len(matching_skills) / len(top_market_stats)) * 100
        if top_market_stats
        else 0.0
    )

    query_context = f"{state.get('role', 'Software Engineer')} in {state.get('country_filter') or 'global market'}"

    analysis = ResumeAnalysis(
        resume_id=resume.id,
        query_context=query_context,
        matching_skills=matching_skills,
        missing_skills=sorted(missing_gaps, key=lambda g: g.market_demand_percentage, reverse=True),
        bonus_skills=bonus_skills[:10],
        gap_score=round(gap_score, 1),
        total_postings_analyzed=total_postings,
        sources=list(set(s for stat in skill_stats_raw[:1] for s in stat.get("sources", []))),
    )

    logger.info(
        "Resume analysis complete",
        extra={
            "matching": len(matching_skills),
            "missing": len(missing_gaps),
            "gap_score": gap_score,
        },
    )

    return {
        "resume_data": resume.model_dump(),
        "resume_analysis": analysis.model_dump(),
        "mcp_calls": mcp_calls,
        "completed_agents": state.get("completed_agents", []) + ["resume_analysis"],
        "messages": [
            AIMessage(
                content=(
                    f"📄 Resume analyzed — **Market Fit Score: {gap_score:.1f}/100**\n"
                    f"✅ Matching: {len(matching_skills)} skills | "
                    f"⚠️ Missing (high priority): {len(analysis.get_high_priority_gaps())}"
                )
            )
        ],
    }
