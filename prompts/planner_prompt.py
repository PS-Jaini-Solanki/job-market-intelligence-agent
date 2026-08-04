"""
prompts/planner_prompt.py
==========================
System prompt for the Planner agent.

The Planner is the "brain" of the system — it receives the user's
natural language query (plus the classifier's extracted facts) and
breaks it down into a structured plan that other agents can execute.

KEY DESIGN DECISIONS:
- We inject the available tools and MCP servers directly into the prompt
  so the LLM knows exactly what's available (no hallucinated tool names).
- We inject the classifier's deterministic facts (role, country, intent,
  technologies) so the LLM cannot forget or guess them.
"""

from __future__ import annotations


PLANNER_SYSTEM_PROMPT = """You are the Planner Agent for a Job Market Intelligence System.

Your job is to analyze a user's question and create a structured execution plan
as a JSON object. Only include tasks that are genuinely needed to answer the
user's question — minimal plans are preferred.

## Available Task Types
- collect_jobs: Collect job postings from the `jobs` MCP (Adzuna + Arbeitnow). Needs: role, country
- analyze_salaries: Parse salary data from collected postings
- extract_skills: Count skill frequencies from job postings
- analyze_trends: Fetch GitHub stats + compute popularity scores. Needs: technologies
- analyze_resume: Parse user resume and compare to market data
- compare_countries: Compare job markets across countries. Needs: country_codes
- generate_recommendations: Create personalized learning roadmap
- generate_report: Compile all findings into a final report

## Available MCP Servers
- jobs: Real-time job postings (Adzuna + Arbeitnow) — for collect_jobs and compare_countries
- github: Repository stats, trending repos — for analyze_trends
- filesystem: Read uploaded resume files, save reports — for analyze_resume and generate_report
- search: Web search via Tavily — for supplementary facts during trend/report tasks
- reddit: Developer community discussions (optional, sentiment only)

## Rules
1. ALWAYS end your plan with a generate_report task.
2. If a resume is uploaded, include analyze_resume.
3. If the query compares countries, include compare_countries.
4. If the query asks about technologies/trends, include analyze_trends with the technologies provided.
5. Keep plans minimal — only include tasks actually needed.
6. Set required_mcp_servers correctly for every task (jobs/github/filesystem/search/reddit).
7. query_type must be one of: job_market | salary | resume_analysis | tech_trend | country_comparison.

## Output Format
Respond with a JSON object following this exact structure:
{
  "tasks": [
    {
      "task_type": "<task_type>",
      "parameters": {"key": "value"},
      "required_mcp_servers": ["<server_name>"],
      "depends_on": []
    }
  ],
  "reasoning": "Brief explanation of why these tasks were chosen",
  "query_type": "<job_market|salary|resume_analysis|tech_trend|country_comparison>",
  "requires_resume": false,
  "estimated_duration_seconds": 30
}"""


def build_planner_prompt(
    user_query: str,
    has_resume: bool = False,
    intent: str = "job_market",
    role: str | None = None,
    country: str | None = None,
    countries: list[str] | None = None,
    technologies: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build the user message for the Planner agent.

    Args:
        user_query: The user's natural language query.
        has_resume: Whether the user has uploaded a resume.
        intent: Classified query intent.
        role: Deterministically extracted role (may be None).
        country: Deterministically extracted country (may be None).
        countries: All deterministically extracted country codes.
        technologies: Deterministically extracted technologies.
        conversation_history: Prior turns, used to resolve follow-ups
            like "what about salaries in India?" after a role was discussed.

    Returns:
        The formatted user message string.
    """
    resume_note = ""
    if has_resume:
        resume_note = "\n[Note: User has uploaded a resume. Include analyze_resume in the plan.]"

    facts = [f"Classified intent: {intent}"]
    if role:
        facts.append(f"Detected role: {role}")
    if country:
        facts.append(f"Detected country: {country}")
    if countries:
        facts.append(f"Detected countries: {', '.join(countries)}")
    if technologies:
        facts.append(f"Detected technologies: {', '.join(technologies)}")
    facts_block = "\n".join(f"- {f}" for f in facts)

    history_block = ""
    if conversation_history:
        lines = [
            f"user: {t.get('content', '')}"
            if t.get("role") == "user"
            else f"assistant: {t.get('content', '')}"
            for t in conversation_history[-6:]
        ]
        history_block = "\nConversation so far (for follow-up context only):\n" + "\n".join(lines)

    return f"""User Query: {user_query}{resume_note}

Deterministic facts extracted from the query (use these — do not guess):
{facts_block}{history_block}

Create a minimal, efficient execution plan for this query.
Only include tasks that are directly necessary to answer the question.
If the query is a follow-up that refers to an earlier question, resolve the
missing context (role/country/technologies) from the conversation above —
do not invent facts not present there.
For compare_countries tasks, set parameters.country_codes to the exact list of
countries the user mentioned (e.g. ["IN", "US"])."""
