"""
agents/planner.py
==================
The Planner Agent — decomposes user queries into execution tasks.

ROLE IN THE SYSTEM:
The Planner is the first agent to run. It receives the user's query
and produces a structured plan (list of AgentTask objects) that tells
other agents exactly what to do.

HOW IT WORKS:
1. Receives user_query from state
2. Calls the LLM with a system prompt describing all available tasks
3. LLM returns a JSON plan (validated by Pydantic)
4. Plan is stored in state.execution_plan
5. Other agents read the plan and skip tasks not assigned to them

This is "task decomposition" — a core pattern in Agentic AI.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.messages import AIMessage

from agents.state import GraphState
from models.agent import AgentTask, MCPServerType, PlannerOutput, TaskType
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Default technologies to analyze if query mentions AI/ML
DEFAULT_AI_TECHNOLOGIES = [
    "Python", "LangChain", "LangGraph", "FastAPI", "Docker",
    "LLMs", "RAG", "MCP", "Hugging Face Transformers", "PyTorch",
    "AWS", "ChromaDB", "PostgreSQL", "Groq",
]


def _extract_role_from_query(query: str) -> str:
    """
    Extract the job role from a user query.

    Uses simple keyword matching — not LLM inference.
    Falls back to "AI Engineer" for ambiguous queries.

    Args:
        query: User's natural language query.

    Returns:
        Job role string.
    """
    query_lower = query.lower()

    # Check for explicitly mentioned roles
    role_keywords = {
        "AI Engineer": ["ai engineer", "ai ml engineer", "artificial intelligence engineer"],
        "Data Scientist": ["data scientist"],
        "ML Engineer": ["ml engineer", "machine learning engineer"],
        "Data Engineer": ["data engineer"],
        "Backend Engineer": ["backend engineer", "backend developer", "software engineer"],
        "Full Stack Engineer": ["full stack", "fullstack"],
        "DevOps Engineer": ["devops", "sre", "site reliability"],
        "GenAI Engineer": ["genai", "gen ai", "generative ai engineer", "llm engineer"],
    }

    for role, keywords in role_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return role

    return "AI Engineer"  # Default role


def _extract_country_from_query(query: str) -> Optional[str]:
    """
    Extract country code from user query.

    Args:
        query: User's natural language query.

    Returns:
        ISO country code string, or None.
    """
    query_lower = query.lower()
    country_map = {
        "IN": ["india", "indian", "bangalore", "mumbai", "delhi"],
        "US": ["usa", "united states", "america", "us market", "american"],
        "GB": ["uk", "united kingdom", "britain", "london"],
        "AU": ["australia", "australian", "sydney", "melbourne"],
        "DE": ["germany", "german", "berlin"],
        "CA": ["canada", "canadian", "toronto"],
        "SG": ["singapore"],
    }

    for code, keywords in country_map.items():
        if any(kw in query_lower for kw in keywords):
            return code

    return None


def _is_comparison_query(query: str) -> bool:
    """Check if query is asking for a country comparison."""
    keywords = ["compare", "vs", "versus", "difference between", "both countries"]
    return any(kw in query.lower() for kw in keywords)


def _is_trend_query(query: str) -> bool:
    """Check if query is asking about technology trends."""
    keywords = ["trending", "trend", "popular", "should i learn", "better", "vs", "comparison"]
    return any(kw in query.lower() for kw in keywords)


def _is_resume_query(query: str) -> bool:
    """Check if query involves resume analysis."""
    keywords = ["my resume", "my skills", "gap", "missing", "what skills am i missing"]
    return any(kw in query.lower() for kw in keywords)


async def run_planner(state: GraphState) -> dict:
    """
    The Planner agent node function.

    LangGraph calls this function with the current state.
    We return a PARTIAL state update (only the fields we're changing).

    Args:
        state: Current graph state.

    Returns:
        Partial state update with execution_plan, role, country_filter.
    """
    user_query = state["user_query"]
    has_resume = bool(state.get("resume_path"))

    logger.info("Planner starting", extra={"query": user_query})

    # --- Step 1: Extract simple facts without LLM ---
    role = state.get("role") or _extract_role_from_query(user_query)
    country = state.get("country_filter") or _extract_country_from_query(user_query)

    # --- Step 2: Call LLM to create the execution plan ---
    llm_service = get_llm_service()
    prompt = build_planner_prompt(user_query, has_resume=has_resume)

    try:
        raw_response = await llm_service.generate(
            user_prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

        # Parse JSON from the LLM response
        plan_dict = _parse_json_from_response(raw_response)

        # Validate with Pydantic
        plan = _build_plan_from_dict(plan_dict, role=role, country=country)

    except Exception as e:
        logger.error("Planner LLM call failed — using fallback plan", extra={"error": str(e)})
        # Fallback: create a sensible default plan
        plan = _create_fallback_plan(
            user_query=user_query,
            role=role,
            country=country,
            has_resume=has_resume,
        )

    logger.info(
        "Planner complete",
        extra={
            "tasks": len(plan.tasks),
            "query_type": plan.query_type,
            "mcp_servers": [s.value for s in plan.get_mcp_servers_needed()],
        },
    )

    # Determine technologies to analyze
    technologies = state.get("technologies_to_analyze") or []
    if not technologies and _is_trend_query(user_query):
        technologies = DEFAULT_AI_TECHNOLOGIES

    return {
        "execution_plan": plan.model_dump(),
        "query_type": plan.query_type,
        "role": role,
        "country_filter": country,
        "technologies_to_analyze": technologies,
        "completed_agents": state.get("completed_agents", []) + ["planner"],
        "messages": [AIMessage(content=f"📋 Plan created: {plan.reasoning}")],
    }


def _parse_json_from_response(response: str) -> dict:
    """
    Extract JSON from an LLM response string.

    LLMs sometimes wrap JSON in markdown code blocks or add extra text.
    This function handles those cases.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed Python dictionary.
    """
    # Try to find JSON in a code block first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to parse the whole response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Find the first { and last } in the response
    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(response[start:end])

    raise ValueError(f"No valid JSON found in response: {response[:200]}")


def _build_plan_from_dict(
    plan_dict: dict,
    role: str,
    country: Optional[str],
) -> PlannerOutput:
    """
    Convert raw plan dictionary to a validated PlannerOutput.

    Adds sensible defaults for missing parameters.

    Args:
        plan_dict: Raw plan dictionary from LLM.
        role: Extracted job role.
        country: Extracted country code.

    Returns:
        Validated PlannerOutput.
    """
    tasks = []
    for task_dict in plan_dict.get("tasks", []):
        # Inject role/country if not present in parameters
        params = task_dict.get("parameters", {})
        if "role" not in params and role:
            params["role"] = role
        if "country" not in params and country:
            params["country"] = country

        # Convert MCP server strings to MCPServerType enums
        mcp_servers = []
        for server_str in task_dict.get("required_mcp_servers", []):
            try:
                mcp_servers.append(MCPServerType(server_str))
            except ValueError:
                pass  # Skip unknown server types

        task = AgentTask(
            task_type=TaskType(task_dict.get("task_type", "generate_report")),
            parameters=params,
            required_mcp_servers=mcp_servers,
        )
        tasks.append(task)

    return PlannerOutput(
        tasks=tasks,
        reasoning=plan_dict.get("reasoning", "Plan generated by LLM"),
        query_type=plan_dict.get("query_type", "job_market"),
        requires_resume=plan_dict.get("requires_resume", False),
        estimated_duration_seconds=plan_dict.get("estimated_duration_seconds"),
    )


def _create_fallback_plan(
    user_query: str,
    role: str,
    country: Optional[str],
    has_resume: bool,
) -> PlannerOutput:
    """
    Create a sensible default plan when LLM planning fails.

    This ensures the system still works even if the LLM is unavailable.

    Args:
        user_query: Original query.
        role: Extracted role.
        country: Extracted country.
        has_resume: Whether user has a resume.

    Returns:
        A default PlannerOutput.
    """
    params = {"role": role}
    if country:
        params["country"] = country

    tasks = [
        AgentTask(
            task_type=TaskType.COLLECT_JOBS,
            parameters=params,
            required_mcp_servers=[],
        ),
        AgentTask(
            task_type=TaskType.EXTRACT_SKILLS,
            parameters=params,
            required_mcp_servers=[],
        ),
        AgentTask(
            task_type=TaskType.ANALYZE_SALARIES,
            parameters=params,
            required_mcp_servers=[],
        ),
    ]

    if has_resume:
        tasks.append(AgentTask(
            task_type=TaskType.ANALYZE_RESUME,
            parameters=params,
            required_mcp_servers=[MCPServerType.FILESYSTEM],
        ))

    tasks.append(AgentTask(
        task_type=TaskType.GENERATE_REPORT,
        parameters=params,
        required_mcp_servers=[],
    ))

    return PlannerOutput(
        tasks=tasks,
        reasoning="Fallback plan (LLM planning unavailable) — running standard job market analysis",
        query_type="job_market",
        requires_resume=has_resume,
    )
