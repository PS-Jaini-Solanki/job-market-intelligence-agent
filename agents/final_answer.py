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
from tools.job_search_tool import (
    COUNTRY_NAMES,
    DEFAULT_ADZUNA_COUNTRY,
    country_name,
)

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
    """The data sources that actually returned postings this turn."""
    collected = (state.get("job_search_results") or {}).get("sources_queried") or []
    if collected:
        return list(collected)
    sources: set[str] = set()
    for stat in state.get("skill_statistics") or []:
        sources.update(stat.get("sources", []))
    return sorted(sources)


def _sources_text(sources: list[str]) -> str:
    return ", ".join(sources) or "no source"


def _location_phrase(state: GraphState) -> str:
    """
    Where the analyzed postings are from, e.g. " in **India**".

    With no country named, the postings' own countries are reported (the
    default board plus any Arbeitnow matches), never "globally".
    """
    country = state.get("country_filter")
    if country:
        return f" in **{country_name(country)}**"
    postings = (state.get("job_search_results") or {}).get("postings") or []
    counts: dict[str, int] = {}
    for posting in postings:
        label = country_name(posting.get("country")) or "location not stated"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return " in " + " and ".join(f"**{label}** ({n})" for label, n in ordered)


def _context_notes(state: GraphState) -> str:
    """
    Trailing notes about what the answer assumed or left out: a country the
    sources can't cover, or no country named at all.
    """
    notes: list[str] = []
    unsupported = state.get("unsupported_countries") or []
    if unsupported:
        one = len(unsupported) == 1
        notes.append(
            f"{', '.join(unsupported)} {'is' if one else 'are'} not covered by my live job "
            f"sources (covered: {', '.join(COUNTRY_NAMES.values())}), so "
            f"{'it is' if one else 'they are'} left out."
        )
    if not state.get("country_filter") and not state.get("countries"):
        notes.append(
            f"You didn't name a country, so this uses the default "
            f"{country_name(DEFAULT_ADZUNA_COUNTRY)} market. Name a country (e.g. India, UK, "
            "Germany) for local results."
        )
    return "".join(f"\n\n_{n}_" for n in notes)


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
    query_ctx = analysis.get("query_context") or (
        f"{role} roles in {country_name(country) or 'all markets searched'}"
    )
    sources = _sources_text(analysis.get("sources") or _sources(state))

    matching = _skill_names(analysis.get("matching_skills"))
    missing = analysis.get("missing_skills") or []

    lines = [
        f"**Your resume vs. {query_ctx}**",
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
    return "\n".join(lines) + _resume_role_note(state)


def _resume_role_note(state: GraphState) -> str:
    """Ask for the target role when the comparison used a default one."""
    if not state.get("role_is_default"):
        return ""
    return (
        "\n\n_You didn't say which role you're targeting, so I compared your resume "
        f"against **{state.get('role')}** postings. Tell me the role (e.g. \"Data "
        "Scientist in India\") for a precise comparison._"
    )


def _compose_resume_no_skills_answer(state: GraphState) -> str:
    """A resume was read but contains no technical skill we recognize."""
    resume_data = state.get("resume_data") or {}
    name = resume_data.get("file_name") or "your resume"
    return (
        f"I read **{name}**, but I couldn't find any technical skills I recognize in it "
        "(languages, frameworks, cloud, databases, tools), so I can't score it against "
        "the market. If it's a technical resume, make sure the skills are written out as "
        "text (not inside images) and try again."
    )


def _resume_fit_footer(state: GraphState) -> str:
    """Short personal-fit addendum for a job-market answer when a resume is on file."""
    analysis = state.get("resume_analysis") or {}
    if not analysis:
        return ""
    missing = [g["skill"]["normalized_name"] for g in (analysis.get("missing_skills") or [])[:5]]
    lines = [
        "",
        f"📄 **Your resume:** Market Fit Score **{analysis.get('gap_score', 0):.1f}/100** "
        "against these postings.",
    ]
    if missing:
        lines.append(f"Top gaps to close: {', '.join(missing)}.")
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
    location = f" in {country_name(country)}" if country else ""
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
        where = f" in {country_name(country)}" if country else ""
        return (
            f"I couldn't find enough live postings for **{role}**{where} right now "
            "to analyze. Try a broader role name, another country, or again in a moment."
            + _role_default_disclaimer(state) + _context_notes(state)
        )

    total = skill_stats[0].get("total_postings", 0)
    lines = [
        f"## Top in-demand skills for **{role}**{_location_phrase(state)}",
        "",
        f"Based on **{total} live job postings** "
        f"({_sources_text(_sources(state))}):",
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
    return (
        "\n".join(lines) + _resume_fit_footer(state)
        + _role_default_disclaimer(state) + _context_notes(state)
    )


def _compose_salary_answer(state: GraphState) -> str:
    """Answer a salary question directly."""
    salary = state.get("salary_data") or {}
    role = state.get("role") or "this role"
    country = state.get("country_filter")
    location = f" in {country_name(country)}" if country else ""
    sample = salary.get("sample_count", 0)
    postings = len((state.get("job_search_results") or {}).get("postings") or [])

    if sample < 1:
        return (
            f"Not enough salary data for **{role}**{location} right now — none of the "
            f"{postings} live postings analyzed disclosed a salary range."
            + _role_default_disclaimer(state) + _context_notes(state)
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

    of_total = f" of {postings}" if postings else ""
    lines.extend(
        ["", f"*{sample}{of_total} postings listed a salary — treat as indicative.*"]
    )
    return "\n".join(lines) + _role_default_disclaimer(state) + _context_notes(state)


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
            "I couldn't gather trend data right now (no GitHub data and no matching "
            "postings). Try naming technologies explicitly, e.g. 'LangGraph vs CrewAI'."
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
    if not state.get("technologies_to_analyze"):
        lines.extend(["", "_You didn't name technologies, so these are the top skills "
                          "found in the analyzed postings._"])
    return "\n".join(lines) + _context_notes(state)


def _compose_comparison_answer(state: GraphState) -> str:
    """Answer a country-comparison question directly."""
    comparison = state.get("country_comparison") or {}
    role = state.get("role") or "Software Engineer"

    if not comparison or "error" in comparison:
        reason = (comparison or {}).get("error") or "live job data wasn't available"
        return f"I couldn't complete the comparison for **{role}**: {reason}" + _context_notes(state)

    countries = comparison.get("countries", {})
    if not countries:
        return "No comparison data available."

    lines = [
        f"## {role} job market comparison",
        "",
        f"| Metric | {' | '.join(country_name(c) for c in countries)} |",
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
    return "\n".join(lines) + _context_notes(state)


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
    resume_data = state.get("resume_data") or {}

    # Resume-fit questions. A resume uploaded earlier stays on file for the
    # whole conversation, so it only shapes the answer when this turn is
    # actually about it; other intents always get their own answer.
    if intent == "resume_analysis":
        if state.get("resume_analysis"):
            return _compose_resume_answer(state)
        if not resume_data:
            return _compose_resume_missing_answer(state)
        if not resume_data.get("extracted_skills"):
            return _compose_resume_no_skills_answer(state)
        return _compose_resume_no_market_data_answer(state)
    if intent == "salary":
        return _compose_salary_answer(state)
    if intent == "tech_trend":
        return _compose_trend_answer(state)
    if intent == "country_comparison":
        return _compose_comparison_answer(state)

    # job_market: answer the question; a resume on file adds a personal-fit line.
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
