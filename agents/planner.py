"""
agents/planner.py
==================
The Planner Agent — decomposes user queries into execution tasks.

ROLE IN THE SYSTEM:
The Planner is the second agent to run (after the QueryClassifier).
It receives the user's query plus the classifier's extracted facts
(role, country, technologies) and produces a structured plan
(list of AgentTask objects) that tells other agents exactly what to do.

HOW IT WORKS:
1. Uses the deterministic role/country/tech extraction from the classifier
2. Calls the LLM with a system prompt describing all available tasks
3. LLM returns a JSON plan (validated by Pydantic)
4. Plan is stored in state.execution_plan
5. Other agents read the plan and execute their assigned tasks

IMPORTANT DESIGN CHANGE:
The planner NEVER guesses a role. If the user didn't mention a job role,
role stays None and the plan reflects that (generic market overview).

This is "task decomposition" — a core pattern in Agentic AI.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.messages import AIMessage

from agents.classifier import (
    ANALYTICAL_INTENTS,
    QueryIntent,
    _extract_country_from_query,
    _extract_role_from_query,
    _extract_technologies_from_query,
)
from agents.state import GraphState
from models.agent import AgentTask, MCPServerType, PlannerOutput, TaskType
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Default technologies to analyze when a trend query mentions no tech.
DEFAULT_AI_TECHNOLOGIES = [
    "Python", "LangChain", "LangGraph", "FastAPI", "Docker",
    "LLMs", "RAG", "MCP", "Hugging Face Transformers", "PyTorch",
    "AWS", "ChromaDB", "PostgreSQL", "Groq",
]

# Matches AI/machine-learning language so "AI job market" defaults to AI Engineer.
_AI_ROLE_RE = re.compile(
    r"\b(ai|gen ?ai|llm|artificial intelligence|machine learning|ml)\b",
    re.IGNORECASE,
)


def _default_role_for_query(query: str) -> str:
    """
    Choose a sensible generic role when the user names no explicit one.

    AI-flavored queries (e.g. "Compare AI job markets in India vs USA")
    default to AI Engineer; everything else defaults to Software Engineer.

    Args:
        query: The user's natural language query.

    Returns:
        A job role string to search job APIs with.
    """
    if _AI_ROLE_RE.search(query):
        return "AI Engineer"
    return "Software Engineer"


def _facts_from_history(conversation_history: list[dict]) -> dict:
    """
    Inherit role/country/technologies from the most recent analytical turn.

    Follow-up questions ("what about salaries in India?") reuse the role that
    was discussed earlier instead of falling back to a generic default.

    Args:
        conversation_history: Prior turns as {"role", "content"} dicts.

    Returns:
        Dict with role, country, countries, technologies (may be empty).
    """
    facts: dict = {"role": None, "country": None, "countries": [], "technologies": []}
    if not conversation_history:
        return facts

    for turn in reversed(conversation_history):
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        prev_query = turn.get("content", "")
        if not prev_query:
            continue
        prev_role = _extract_role_from_query(prev_query)
        prev_country = _extract_country_from_query(prev_query)
        prev_tech = _extract_technologies_from_query(prev_query)
        if prev_role and not facts["role"]:
            facts["role"] = prev_role
        if prev_country and not facts["country"]:
            facts["country"] = prev_country
        if prev_tech and not facts["technologies"]:
            facts["technologies"] = prev_tech
        if facts["role"] and facts["country"] and facts["technologies"]:
            break
    return facts


async def run_planner(state: GraphState) -> dict:
    """
    The Planner agent node function.

    Args:
        state: Current graph state (includes classifier output).

    Returns:
        Partial state update with execution_plan, role, country_filter.
    """
    user_query = state["user_query"]
    has_resume = bool(state.get("resume_path"))
    intent = state.get("query_intent") or QueryIntent.JOB_MARKET.value
    conversation_history = state.get("conversation_history") or []

    logger.info("Planner starting", extra={"query": user_query, "intent": intent})

    # --- Step 1: Extract simple facts deterministically (no LLM) -------
    role = state.get("role") or _extract_role_from_query(user_query)
    country = state.get("country_filter") or _extract_country_from_query(user_query)
    countries = state.get("countries") or []

    # For follow-up questions, inherit facts from the previous analytical turn
    # so "what about salaries in India?" keeps the earlier role (e.g. DevOps).
    inherited = _facts_from_history(conversation_history)
    if not role:
        role = inherited["role"]
    if not country:
        country = inherited["country"]
    if not countries:
        countries = state.get("countries") or []

    # For genuine analytical queries with no explicit role, use a sensible
    # generic market default — this is a real job-market question, not an
    # ambiguous one, so a concrete role is required to search APIs.
    if not role and intent in ANALYTICAL_INTENTS:
        role = _default_role_for_query(user_query)

    technologies = state.get("technologies_to_analyze") or []
    if not technologies:
        technologies = _extract_technologies_from_query(user_query)
    # Only inherit technologies from an earlier turn when THIS query has no
    # role of its own. A query naming its own role (e.g. "what technology
    # should I learn for a Cloud Engineer?") is a brand-new, self-sufficient
    # question — inheriting a leftover technology name from an unrelated
    # earlier question (e.g. "should I learn Python?") caused a spurious,
    # unrelated Technology Trends section to appear in an otherwise-correct
    # job_market report. This mirrors the same guard in agents/classifier.py
    # (this is a second, independent place the same leak could happen from,
    # since the planner re-derives facts from conversation_history itself
    # rather than trusting the classifier's already-fixed output).
    if not technologies and not _extract_role_from_query(user_query):
        technologies = inherited["technologies"]
    if not technologies and intent == QueryIntent.TECH_TREND.value:
        technologies = DEFAULT_AI_TECHNOLOGIES

    # --- Step 2: Call LLM to create the execution plan -----------------
    llm_service = get_llm_service()
    prompt = build_planner_prompt(
        user_query,
        has_resume=has_resume,
        intent=intent,
        role=role,
        country=country,
        countries=countries,
        technologies=technologies,
        conversation_history=conversation_history,
    )

    plan: Optional[PlannerOutput] = None
    try:
        raw_response = await llm_service.generate(
            user_prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        plan_dict = _parse_json_from_response(raw_response)
        plan = _build_plan_from_dict(
            plan_dict,
            role=role,
            country=country,
            countries=countries,
            intent=intent,
            technologies=technologies,
        )
    except Exception as e:
        logger.error("Planner LLM call failed — using fallback plan", extra={"error": str(e)})

    if plan is None or not plan.tasks:
        plan = _create_fallback_plan(
            user_query=user_query,
            role=role,
            country=country,
            countries=countries,
            has_resume=has_resume,
            intent=intent,
            technologies=technologies,
        )

    # If even the fallback produced no work (shouldn't happen for
    # analytical intents), stop gracefully and let the ChatResponder explain.
    should_stop = not plan.tasks

    logger.info(
        "Planner complete",
        extra={
            "tasks": len(plan.tasks),
            "intent": intent,
            "role": role,
            "country": country,
            "query_type": plan.query_type,
            "mcp_servers": [s.value for s in plan.get_mcp_servers_needed()],
        },
    )

    return {
        "execution_plan": plan.model_dump(),
        "query_type": plan.query_type,
        "role": role,
        "country_filter": country,
        "countries": countries,
        "technologies_to_analyze": technologies,
        "should_stop": should_stop,
        "completed_agents": state.get("completed_agents", []) + ["planner"],
        "messages": [AIMessage(content=f"📋 Plan created: {plan.reasoning}")],
    }


# ============================================================
# Parsing helpers
# ============================================================

def _parse_json_from_response(response: str) -> dict:
    """
    Extract JSON from an LLM response string.

    Handles markdown code fences and surrounding text.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed Python dictionary.

    Raises:
        ValueError: If no valid JSON can be found.
    """
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(response[start:end])

    raise ValueError(f"No valid JSON found in response: {response[:200]}")


def _build_plan_from_dict(
    plan_dict: dict,
    role: Optional[str],
    country: Optional[str],
    countries: list[str],
    intent: str,
    technologies: list[str],
) -> PlannerOutput:
    """
    Convert raw plan dictionary to a validated PlannerOutput.

    Injects the deterministic role/country/tech facts where missing.

    Args:
        plan_dict: Raw plan dictionary from LLM.
        role: Extracted job role.
        country: Extracted country code.
        countries: All extracted country codes.
        intent: Classified query intent.
        technologies: Extracted technologies.

    Returns:
        Validated PlannerOutput.
    """
    tasks = []
    for task_dict in plan_dict.get("tasks", []):
        params = dict(task_dict.get("parameters", {}))

        task_type = task_dict.get("task_type", "generate_report")

        # Inject facts the LLM may have omitted.
        if task_type in ("collect_jobs", "compare_countries") and "role" not in params and role:
            params["role"] = role
        if task_type in ("collect_jobs", "compare_countries") and "country" not in params and country:
            params["country"] = country
        if task_type == "compare_countries" and "country_codes" not in params and countries:
            params["country_codes"] = countries
        if task_type == "analyze_trends" and "technologies" not in params and technologies:
            params["technologies"] = technologies

        # Convert MCP server strings to enums.
        mcp_servers = []
        for server_str in task_dict.get("required_mcp_servers", []):
            try:
                mcp_servers.append(MCPServerType(server_str))
            except ValueError:
                pass

        # Auto-annotate MCP servers for known task types so the plan
        # always reflects reality even if the LLM omits them.
        if not mcp_servers:
            mcp_servers = _default_mcp_servers_for_task(task_type)

        tasks.append(AgentTask(
            task_type=TaskType(task_type),
            parameters=params,
            required_mcp_servers=mcp_servers,
        ))

    return PlannerOutput(
        tasks=tasks,
        reasoning=plan_dict.get("reasoning", "Plan generated by LLM"),
        query_type=plan_dict.get("query_type", intent),
        requires_resume=plan_dict.get("requires_resume", False),
        estimated_duration_seconds=plan_dict.get("estimated_duration_seconds"),
    )


def _default_mcp_servers_for_task(task_type: str) -> list[MCPServerType]:
    """
    Return the MCP servers a task type always needs.

    Guarantees the plan's required_mcp_servers always reflect reality.

    Args:
        task_type: The task type string.

    Returns:
        List of MCPServerType values.
    """
    mapping = {
        "collect_jobs": [MCPServerType.JOBS],
        "compare_countries": [MCPServerType.JOBS],
        "analyze_trends": [MCPServerType.GITHUB, MCPServerType.SEARCH],
        "analyze_resume": [MCPServerType.FILESYSTEM],
        "generate_report": [MCPServerType.FILESYSTEM],
    }
    return mapping.get(task_type, [])


# ============================================================
# Fallback plan (used when the LLM is unavailable)
# ============================================================

def _create_fallback_plan(
    user_query: str,
    role: Optional[str],
    country: Optional[str],
    countries: list[str],
    has_resume: bool,
    intent: str,
    technologies: list[str],
) -> PlannerOutput:
    """
    Create a sensible default plan when LLM planning fails.

    Args:
        user_query: Original query.
        role: Extracted role (may be None).
        country: Extracted country (may be None).
        countries: All extracted country codes.
        has_resume: Whether a resume is uploaded.
        intent: Classified intent.
        technologies: Technologies to analyze.

    Returns:
        A default PlannerOutput.
    """
    params: dict = {}
    if role:
        params["role"] = role
    if country:
        params["country"] = country

    tasks: list[AgentTask] = []
    query_type = intent

    if intent == QueryIntent.SALARY.value and role:
        tasks.append(AgentTask(
            task_type=TaskType.COLLECT_JOBS,
            parameters={**params, "focus": "salary"},
            required_mcp_servers=[MCPServerType.JOBS],
        ))
        tasks.append(AgentTask(
            task_type=TaskType.ANALYZE_SALARIES,
            parameters=params,
        ))
    elif intent == QueryIntent.TECH_TREND.value and technologies:
        tasks.append(AgentTask(
            task_type=TaskType.COLLECT_JOBS,
            parameters=params,
            required_mcp_servers=[MCPServerType.JOBS],
        ))
        tasks.append(AgentTask(
            task_type=TaskType.ANALYZE_TRENDS,
            parameters={**params, "technologies": technologies},
            required_mcp_servers=[MCPServerType.GITHUB],
        ))
    elif intent == QueryIntent.COUNTRY_COMPARISON.value:
        tasks.append(AgentTask(
            task_type=TaskType.COLLECT_JOBS,
            parameters=params,
            required_mcp_servers=[MCPServerType.JOBS],
        ))
        compare_params = {**params, "country_codes": countries} if countries else params
        tasks.append(AgentTask(
            task_type=TaskType.COMPARE_COUNTRIES,
            parameters=compare_params,
            required_mcp_servers=[MCPServerType.JOBS],
        ))
    elif intent in ANALYTICAL_INTENTS:
        tasks.append(AgentTask(
            task_type=TaskType.COLLECT_JOBS,
            parameters=params,
            required_mcp_servers=[MCPServerType.JOBS],
        ))
        tasks.append(AgentTask(
            task_type=TaskType.EXTRACT_SKILLS,
            parameters=params,
        ))
        if technologies:
            tasks.append(AgentTask(
                task_type=TaskType.ANALYZE_TRENDS,
                parameters={**params, "technologies": technologies},
                required_mcp_servers=[MCPServerType.GITHUB],
            ))

    if has_resume:
        tasks.append(AgentTask(
            task_type=TaskType.ANALYZE_RESUME,
            parameters=params,
            required_mcp_servers=[MCPServerType.FILESYSTEM],
        ))

    if tasks:
        tasks.append(AgentTask(
            task_type=TaskType.GENERATE_REPORT,
            parameters=params,
            required_mcp_servers=[MCPServerType.FILESYSTEM],
        ))

    return PlannerOutput(
        tasks=tasks,
        reasoning=(
            "Fallback plan (LLM planning unavailable) — "
            f"running standard analysis for intent '{intent}'"
        ),
        query_type=query_type,
        requires_resume=has_resume,
    )
