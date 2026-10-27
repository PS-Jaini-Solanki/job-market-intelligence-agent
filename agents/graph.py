"""
agents/graph.py
===============
LangGraph execution graph — wires all agents together.

HOW LANGGRAPH WORKS:
1. You define a StateGraph with a shared state type.
2. You add nodes (agents) to the graph.
3. You add edges (connections) between nodes.
4. LangGraph executes the graph, calling each node in order.
5. Each node can conditionally route to different next nodes.

GRAPH FLOW:
START → query_classifier
       ├── analytical intent → planner → market_collection → skill_extraction
       │        → trend_analysis (conditional) → resume_analysis (conditional)
       │        → comparison_agent (conditional) → recommendation
       │        → final_answer → report_generation → END
       └── conversational intent / unsupported country → chat_responder → END

The QueryClassifier decides between the two families.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.classifier import is_analytical
from agents.chat_responder import run_chat_responder
from agents.classifier import run_query_classifier
from agents.country_comparison import run_country_comparison
from agents.final_answer import run_final_answer
from agents.market_collection import run_market_collection
from agents.planner import run_planner
from agents.recommendation import run_recommendation
from agents.report_generation import run_report_generation
from agents.resume_analysis import run_resume_analysis
from agents.skill_extraction import run_skill_extraction
from agents.trend_analysis import run_trend_analysis
from agents.state import GraphState

logger = logging.getLogger(__name__)


# ============================================================
# Conditional routing functions
# ============================================================


def route_after_classifier(state: GraphState) -> Literal["planner", "chat_responder"]:
    """
    Route to the analytical pipeline or the direct chat responder.

    - Analytical (job_market, salary, trends, comparison, resume) → planner
    - Conversational (general_chat, out_of_scope) → chat_responder
    - Analytical, but every country named is one the job sources can't
      cover (e.g. only "Japan") → chat_responder, which says so honestly
      instead of answering with another country's data.
    """
    if not is_analytical(state.get("query_intent")):
        return "chat_responder"
    if state.get("unsupported_countries") and not state.get("countries"):
        return "chat_responder"
    return "planner"


def route_after_planner(state: GraphState) -> Literal["market_collection", "chat_responder"]:
    """
    After planning, either start collecting data or hand off to chat.

    If the planner could not build a valid plan (should_stop), the
    ChatResponder explains gracefully instead of silently failing.
    """
    if state.get("should_stop"):
        logger.warning("Graph stopping planning due to invalid/empty plan")
        return "chat_responder"
    return "market_collection"


def _plan_task_types(state: GraphState) -> set[str]:
    """Task types in the current execution plan."""
    plan = state.get("execution_plan") or {}
    return {t.get("task_type") for t in plan.get("tasks", [])}


def _has_compare_countries_task(state: GraphState) -> bool:
    """
    True if a country comparison is needed: the plan says so, or the
    deterministic classifier did (routing never depends on the LLM alone).
    """
    return (
        "compare_countries" in _plan_task_types(state)
        or state.get("query_intent") == "country_comparison"
    )


# Intents where running resume analysis actually serves the question.
# A resume uploaded earlier stays in the session for later turns, so
# `resume_path` alone must never trigger a gap analysis: "what's the
# average salary in Germany?" after uploading a resume is a salary
# question. job_market is included because "what skills do I need for X"
# with a resume on file is answered with the user's personal gaps too.
_RESUME_RELEVANT_INTENTS = {"resume_analysis", "job_market"}


def _wants_resume_analysis(state: GraphState) -> bool:
    """Single gate for resume analysis, used by every route into it."""
    return bool(state.get("resume_path")) and state.get("query_intent") in _RESUME_RELEVANT_INTENTS


def should_analyze_resume(
    state: GraphState,
) -> Literal["resume_analysis", "comparison_agent", "recommendation"]:
    """
    Pick the next analysis step after skills (and trends, if they ran).

    Resume analysis runs only when a resume is on file AND this turn's
    intent is resume-relevant. Otherwise a pending compare_countries task
    goes to the comparison agent so it is never skipped.
    """
    if _wants_resume_analysis(state):
        return "resume_analysis"
    if _has_compare_countries_task(state):
        return "comparison_agent"
    return "recommendation"


def should_analyze_trends(
    state: GraphState,
) -> Literal["trend_analysis", "resume_analysis", "comparison_agent", "recommendation"]:
    """
    Run trend analysis for trend questions (plan task, named technologies,
    or tech_trend intent); otherwise continue with should_analyze_resume.
    """
    if (
        "analyze_trends" in _plan_task_types(state)
        or state.get("technologies_to_analyze")
        or state.get("query_intent") == "tech_trend"
    ):
        return "trend_analysis"
    return should_analyze_resume(state)


def should_run_country_comparison(
    state: GraphState,
) -> Literal["comparison_agent", "recommendation"]:
    """
    Decide whether a dedicated country comparison step is needed.

    Only used when the plan explicitly contains a compare_countries task.
    """
    if _has_compare_countries_task(state):
        return "comparison_agent"
    return "recommendation"


# ============================================================
# Graph construction
# ============================================================


def build_graph() -> StateGraph:
    """
    Build and compile the multi-agent LangGraph.

    Returns:
        A compiled LangGraph StateGraph ready for execution.

    Usage:
        graph = build_graph()
        result = await graph.ainvoke(initial_state)
    """
    graph = StateGraph(GraphState)

    # --- Add all agent nodes ---
    graph.add_node("query_classifier", run_query_classifier)
    graph.add_node("chat_responder", run_chat_responder)
    graph.add_node("planner", run_planner)
    graph.add_node("market_collection", run_market_collection)
    graph.add_node("skill_extraction", run_skill_extraction)
    graph.add_node("trend_analysis", run_trend_analysis)
    graph.add_node("resume_analysis", run_resume_analysis)
    graph.add_node("comparison_agent", run_country_comparison)
    graph.add_node("recommendation", run_recommendation)
    graph.add_node("final_answer", run_final_answer)
    graph.add_node("report_generation", run_report_generation)

    # --- Edges ---
    # START → classifier → (planner | chat_responder)
    graph.add_edge(START, "query_classifier")
    graph.add_conditional_edges(
        "query_classifier",
        route_after_classifier,
        {"planner": "planner", "chat_responder": "chat_responder"},
    )

    # chat_responder → END (direct answer, no report)
    graph.add_edge("chat_responder", END)

    # planner → market_collection (or chat_responder if no valid plan)
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"market_collection": "market_collection", "chat_responder": "chat_responder"},
    )

    # market_collection → skill_extraction (always)
    graph.add_edge("market_collection", "skill_extraction")

    # skill_extraction → trend_analysis OR resume/comparison/recommendation
    # (the same resume gate applies whether or not trends ran)
    graph.add_conditional_edges(
        "skill_extraction",
        should_analyze_trends,
        {
            "trend_analysis": "trend_analysis",
            "resume_analysis": "resume_analysis",
            "comparison_agent": "comparison_agent",
            "recommendation": "recommendation",
        },
    )

    # trend_analysis → resume_analysis OR comparison_agent OR recommendation (conditional)
    graph.add_conditional_edges(
        "trend_analysis",
        should_analyze_resume,
        {
            "resume_analysis": "resume_analysis",
            "comparison_agent": "comparison_agent",
            "recommendation": "recommendation",
        },
    )

    # resume_analysis → comparison_agent OR recommendation (conditional)
    graph.add_conditional_edges(
        "resume_analysis",
        should_run_country_comparison,
        {
            "comparison_agent": "comparison_agent",
            "recommendation": "recommendation",
        },
    )

    # comparison_agent → recommendation (always)
    graph.add_edge("comparison_agent", "recommendation")

    # recommendation → final_answer → report_generation (always)
    # The FinalAnswer agent writes the concise question-aligned answer first;
    # ReportGeneration then persists the full report to disk for the record.
    graph.add_edge("recommendation", "final_answer")
    graph.add_edge("final_answer", "report_generation")

    # report_generation → END (always)
    graph.add_edge("report_generation", END)

    # Compile the graph
    compiled = graph.compile()

    logger.info(
        "LangGraph compiled",
        extra={"nodes": list(compiled.get_graph().nodes.keys())},
    )

    return compiled


# Module-level compiled graph singleton
_compiled_graph = None


def get_graph():
    """
    Return the shared compiled graph (creates it on first call).
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
