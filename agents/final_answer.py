"""
agents/final_answer.py
=======================
Final Answer Agent — composes the concise, question-aligned answer.

WHY IT EXISTS
-------------
Analytical queries used to end with a big multi-section report regardless of
what was actually asked. Users got a wall of tables even for simple questions
like "what skills am I lacking for a DevOps role?".

This agent reads ONLY the evidence relevant to the user's question and writes
a short, direct `direct_answer` into state. It is fully rule-based (no LLM),
so the answer is deterministic, always available, and never invents numbers.

The full report is still built and saved to disk by ReportGeneration for
persistence, but the UI surfaces this concise answer as the primary response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agents.state import GraphState

logger = logging.getLogger(__name__)


# ============================================================
# Formatting helpers
# ============================================================

def _fmt_money(value: float | None, currency: str = "USD") -> str:
    """Format a salary value like 'USD 117,881'."""
    if value is None:
        return "N/A"
    return f"{currency} {value:,.0f}"


def _skill_names(items: list[dict]) -> list[str]:
    """Extract normalized skill names from serialized Skill/SkillGap dicts."""
    names = []
    for item in items or []:
        skill = item.get("skill", item) if isinstance(item, dict) else item
        name = skill.get("normalized_name") if isinstance(skill, dict) else None
        if name:
            names.append(name)
    return names


def _sources(state: GraphState) -> list[str]:
    """Collect the data sources recorded in state."""
    sources: set[str] = set()
    for stat in state.get("skill_statistics") or []:
        for s in stat.get("sources", []):
            sources.add(s)
    return sorted(sources) or ["adzuna", "arbeitnow"]


def _role_default_disclaimer(state: GraphState) -> str:
    """
    A trailing note shown whenever the answer's role came from the
    Planner's generic fallback default rather than something the user
    actually said. Without this, a user who never mentioned a role saw
    "Software Engineer" quoted back at them as if they'd asked about it —
    reading as the system being hardcoded to one role rather than a
    dynamic one that adapts to whatever role is actually asked about.
    """
    if not state.get("role_is_default"):
        return ""
    role = state.get("role") or "Software Engineer"
    return (
        f"\n\n_You didn't mention a specific role, so this is a general **{role}** "
        'market overview — ask about a specific role (e.g. "Data Scientist", '
        '"DevOps Engineer", "Cloud Engineer") to get results for that role instead._'
    )


# ============================================================
# Per-question composers
# ============================================================

def _compose_resume_answer(state: GraphState) -> str:
    """Answer a resume-fit / missing-skills question directly."""
    analysis = state.get("resume_analysis") or {}
    role = state.get("role") or "Software Engineer"
    country = state.get("country_filter")
    recommendations = state.get("recommendations") or []

    fit = analysis.get("gap_score")
    total = analysis.get("total_postings_analyzed") or 0
    query_ctx = analysis.get("query_context") or f"{role} in {country or 'global market'}"
    sources = ", ".join(analysis.get("sources") or ["adzuna", "arbeitnow"])

    matching = _skill_names(analysis.get("matching_skills"))
    missing = analysis.get("missing_skills") or []

    lines = [
        f"**Your skills vs. the {query_ctx}**",
        "",
        f"**Market Fit Score: {fit:.1f}/100** "
        f"(based on {total} job postings from {sources}).",
        "",
        f"✅ **You already match ({len(matching)}):** "
        f"{', '.join(matching[:12]) or 'none found'}",
    ]

    if missing:
        lines.extend([
            "",
            "⚠️ **Skills the market wants that you're missing** "
            "(priority by demand):",
            "",
        ])
        for gap in missing[:8]:
            skill_name = gap["skill"]["normalized_name"]
            pct = gap.get("market_demand_percentage", 0)
            priority = gap.get("priority", "LOW")
            lines.append(
                f"- **{skill_name}** — in {pct:.1f}% of postings ({priority} priority)"
            )
        if len(missing) > 8:
            lines.append(f"- ...and {len(missing) - 8} more.")
    else:
        lines.extend([
            "",
            "✅ **No missing market skills detected — great coverage!**",
        ])

    if recommendations:
        lines.extend(["", "💡 **Next step:**", ""])
        lines.extend(f"- {rec}" for rec in recommendations[:3])

    lines.append("\n*Answers are based on live job postings only.*")
    return "\n".join(lines)


def _compose_resume_no_market_data_answer(state: GraphState) -> str:
    """
    Answer a resume question when the resume was parsed but there was no
    market data to compare it against.

    Without this branch, a resume-analysis request with no job postings
    available fell through to the generic job-market "couldn't find enough
    postings" message, which never even acknowledged the resume was read —
    confusing for a user who just uploaded a file.
    """
    resume_data = state.get("resume_data") or {}
    role = state.get("role") or "this role"
    country = state.get("country_filter")
    location = f" in {country}" if country else ""
    skills = resume_data.get("extracted_skills") or []
    skill_names = [
        s.get("normalized_name") for s in skills if isinstance(s, dict) and s.get("normalized_name")
    ]

    lines = [
        f"I read your resume and found {len(skill_names)} skills"
        + (f" ({', '.join(skill_names[:10])})" if skill_names else "")
        + ".",
        "",
        f"I couldn't pull enough live job postings for **{role}**{location} right now "
        "to compare it against, so I can't score your market fit or list missing skills "
        "yet. Please try again in a moment.",
    ]
    return "\n".join(lines)


def _compose_resume_missing_answer(state: GraphState) -> str:
    """
    Answer a resume question when no resume was ever attached.

    Distinct from `_compose_resume_no_market_data_answer`, which handles
    "resume was read but no market data to compare it against". This
    handles the case where there's no resume_data at all — no file was
    attached, or attaching it failed silently.
    """
    role = state.get("role")
    role_is_default = bool(state.get("role_is_default"))
    # Don't mention a role the user never actually asked about — a
    # Planner-guessed default has no place in a message that's purely
    # about a missing file.
    role_hint = f" for a **{role}** role" if role and not role_is_default else ""
    return (
        "I don't see a resume attached to this conversation yet"
        f"{role_hint}. Please attach a PDF, DOCX, or TXT file with your "
        "next message and I'll compare it against real job postings."
    )


def _compose_job_market_answer(state: GraphState) -> str:
    """Answer 'what skills are in demand for X?' directly."""
    role = state.get("role") or "Software Engineer"
    country = state.get("country_filter")
    skill_stats = state.get("skill_statistics") or []
    recommendations = state.get("recommendations") or []

    if not skill_stats:
        return (
            f"I couldn't find enough live postings for **{role}** "
            f"{'in ' + country if country else 'in the global market'} right now. "
            "Please try again in a moment." + _role_default_disclaimer(state)
        )

    total = skill_stats[0].get("total_postings", 0)
    location = f" in **{country}**" if country else " globally"
    lines = [
        f"## Top in-demand skills for **{role}**{location}",
        "",
        f"Based on **{total} live job postings** "
        f"({', '.join(_sources(state))}):",
        "",
    ]
    for stat in skill_stats[:8]:
        lines.append(
            f"- **{stat['skill_name']}** — {stat['occurrence_count']}/{total} "
            f"postings ({stat['percentage']}%)"
        )

    if recommendations:
        lines.extend(["", "💡 **Prioritize:**", ""])
        lines.extend(f"- {rec}" for rec in recommendations[:3])

    lines.extend(
        ["", "*Percentages are of the analyzed postings only — not the whole market.*"]
    )
    return "\n".join(lines) + _role_default_disclaimer(state)


def _compose_salary_answer(state: GraphState) -> str:
    """Answer a salary question directly."""
    salary = state.get("salary_data") or {}
    role = state.get("role") or "this role"
    country = state.get("country_filter")
    location = f" in {country}" if country else ""
    sample = salary.get("sample_count", 0)

    if sample < 1:
        return (
            f"Not enough salary data for **{role}**{location} right now — "
            "fewer than a handful of live postings disclosed compensation."
            + _role_default_disclaimer(state)
        )

    currency = salary.get("currency", "USD")
    lines = [
        f"## Salary for **{role}**{location}",
        "",
        f"Based on **{sample} live postings** that disclosed a range:",
        "",
    ]
    if salary.get("median_salary") is not None:
        lines.append(
            f"- **Median:** {_fmt_money(salary.get('median_salary'), currency)}\n"
            f"- **Range:** {_fmt_money(salary.get('min_salary'), currency)} – "
            f"{_fmt_money(salary.get('max_salary'), currency)}"
        )
    else:
        lines.append(
            f"- **Range:** {_fmt_money(salary.get('min_salary'), currency)} – "
            f"{_fmt_money(salary.get('max_salary'), currency)}"
        )

    lines.extend(
        ["", "*Only a fraction of postings list salaries — treat as indicative.*"]
    )
    return "\n".join(lines) + _role_default_disclaimer(state)


def _compose_trend_answer(state: GraphState) -> str:
    """Answer a technology-trend question directly."""
    trend_data = state.get("trend_data") or {}
    trends = sorted(
        trend_data.get("trends", []),
        key=lambda t: t.get("popularity_score", 0),
        reverse=True,
    )
    role = state.get("role") or "Software Engineer"

    if not trends:
        return (
            f"I couldn't gather trend data for the technologies you asked about. "
            "Try naming technologies explicitly (e.g. 'LangGraph', 'MCP')."
        )

    lines = [
        "## Technology trends",
        "",
        "Popularity score = GitHub activity + job-market demand "
        f"(for **{role}**):",
        "",
    ]
    for trend in trends[:8]:
        tech = trend.get("technology", "?")
        score = trend.get("popularity_score", 0)
        job_pct = trend.get("job_demand_percentage", 0)
        lines.append(
            f"- **{tech}** — score {score:.1f} "
            f"(job demand {job_pct:.1f}%)"
        )
    return "\n".join(lines)


def _compose_comparison_answer(state: GraphState) -> str:
    """Answer a country-comparison question directly."""
    comparison = state.get("country_comparison") or {}
    role = state.get("role") or "Software Engineer"

    if not comparison or "error" in comparison:
        return (
            f"I couldn't complete the comparison for **{role}** "
            "— live job data for the requested countries wasn't available."
        )

    countries = comparison.get("countries", {})
    if not countries:
        return "No comparison data available."

    lines = [
        f"## {role} job market comparison",
        "",
        f"| Metric | {' | '.join(countries)} |",
        "|--------|" + "------|" * len(countries),
    ]

    volume = [str(v.get("posting_count", 0)) for v in countries.values()]
    lines.append("| **Postings found** | " + " | ".join(volume) + " |")

    salary_cells = []
    for v in countries.values():
        sal = v.get("salary", {})
        if sal.get("median"):
            salary_cells.append(
                f"{_fmt_money(sal['median'], sal.get('currency', 'USD'))}"
            )
        else:
            salary_cells.append("N/A")
    lines.append("| **Median salary** | " + " | ".join(salary_cells) + " |")

    top_skill_cells = []
    for v in countries.values():
        top = [s["skill"] for s in v.get("top_skills", [])[:3]]
        top_skill_cells.append(", ".join(top) if top else "N/A")
    lines.append("| **Top skills** | " + " | ".join(top_skill_cells) + " |")

    insights = comparison.get("cross_country_insights", {})
    common = insights.get("common_skills", [])
    if common:
        lines.extend(["", f"**Shared across markets:** {', '.join(common[:6])}"])

    lines.extend(
        ["", "*Country coverage reflects our job sources, not true market size.*"]
    )
    return "\n".join(lines)


# ============================================================
# Entry point
# ============================================================

def compose_final_answer(state: GraphState) -> str:
    """
    Build the concise, question-aligned answer for the current state.

    Picks the composer based on the intent and the evidence actually
    available in state. Pure rules + real data — no LLM, no invention.

    Args:
        state: Current graph state.

    Returns:
        A short Markdown answer string.
    """
    intent = state.get("query_intent") or "job_market"
    has_resume_analysis = bool(state.get("resume_analysis"))
    has_resume_data = bool(state.get("resume_data"))

    # Resume-fit questions take priority whenever resume analysis ran —
    # but ONLY for intents where the resume is actually the point of the
    # question. A resume uploaded earlier in the conversation stays in
    # session state for the rest of the chat (so follow-ups don't need
    # a re-upload), and `should_analyze_resume` in agents/graph.py can
    # still run resume_analysis on later turns. Without this intent
    # restriction, asking a completely unrelated question later — e.g.
    # "what's the average salary for a Backend Engineer in Germany?"
    # (intent=salary) — would silently return a stale resume-fit report
    # instead of answering the salary question at all. `job_market` is
    # kept because "here's my resume, what skills do I need for X role"
    # is a legitimate combined question; salary/tech_trend/comparison
    # intents are not resume questions and must always get their own
    # answer.
    if has_resume_analysis and intent in ("resume_analysis", "job_market"):
        return _compose_resume_answer(state)
    # Resume was uploaded and parsed, but there was no market data to compare
    # it against — acknowledge the resume instead of giving a generic
    # job-market "no postings" message that never mentions it.
    if intent == "resume_analysis" and has_resume_data and not has_resume_analysis:
        return _compose_resume_no_market_data_answer(state)
    # The user asked a resume question but no resume was ever actually
    # attached to this conversation (resume_data is only set once a file
    # is successfully parsed). Without this branch, this fell all the way
    # through to the generic job-market composer, which silently answered
    # with whatever role/country happened to be in state — producing an
    # answer that looked plausible but had nothing to do with the resume
    # request and never told the user a file was missing.
    if intent == "resume_analysis" and not has_resume_data:
        return _compose_resume_missing_answer(state)
    if intent == "salary":
        return _compose_salary_answer(state)
    if intent == "tech_trend":
        return _compose_trend_answer(state)
    if intent == "country_comparison":
        return _compose_comparison_answer(state)

    return _compose_job_market_answer(state)


async def run_final_answer(state: GraphState) -> dict:
    """
    Final Answer agent node function.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with direct_answer populated.
    """
    answer = compose_final_answer(state)
    logger.info(
        "Final answer composed",
        extra={"intent": state.get("query_intent"), "length": len(answer)},
    )
    return {
        "direct_answer": answer,
        "completed_agents": state.get("completed_agents", []) + ["final_answer"],
        "messages": [AIMessage(content=answer)],
    }
