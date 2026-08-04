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
       │        → recommendation → report_generation → END
       └── conversational intent → chat_responder → END

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

    The classifier's intent decides:
    - Analytical (job_market, salary, trends, comparison, resume) → planner
    - Conversational (general_chat, out_of_scope) → chat_responder
    """
    intent = state.get("query_intent")
    if is_analytical(intent):
        return "planner"
    return "chat_responder"


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


def should_analyze_trends(
    state: GraphState,
) -> Literal["trend_analysis", "resume_analysis"]:
    """
    Decide whether to run trend analysis based on the plan.

    Runs trend analysis if the plan includes an analyze_trends task or
    technologies are present in state.
    """
    plan = state.get("execution_plan", {})
    tasks = plan.get("tasks", [])
    technologies = state.get("technologies_to_analyze", [])

    has_trend_task = any(t.get("task_type") == "analyze_trends" for t in tasks)

    if has_trend_task or technologies:
        return "trend_analysis"
    return "resume_analysis"


def _has_compare_countries_task(state: GraphState) -> bool:
    """True if the plan contains a compare_countries task."""
    plan = state.get("execution_plan", {})
    tasks = plan.get("tasks", [])
    return any(t.get("task_type") == "compare_countries" for t in tasks)



# Intents where running resume analysis actually serves the question.
# A resume uploaded earlier in the conversation stays in session state
# for the rest of the chat (see chainlit_ui/app.py), so `resume_path`
# can still be set on turns that have nothing to do with the resume —
# e.g. "what's the average salary in Germany?" right after uploading a
# resume for an India role. Running (and expensively LLM-scoring) a
# resume gap analysis on every later turn regardless of relevance was
# both wasted work and, until final_answer.py was fixed to also check
# intent, the reason unrelated questions got a stale resume report
# instead of an answer.
_RESUME_RELEVANT_INTENTS = {"resume_analysis", "job_market"}


def should_analyze_resume(
    state: GraphState,
) -> Literal["resume_analysis", "comparison_agent", "recommendation"]:
    """
    Decide whether to run resume analysis based on the plan.

    Runs resume analysis only if a resume_path exists AND the current
    turn's intent is actually resume-relevant. Otherwise, if a
    compare_countries task is pending, route to the comparison agent so
    it is never skipped by the trend-analysis path.
    """
    if state.get("resume_path") and state.get("query_intent") in _RESUME_RELEVANT_INTENTS:
        return "resume_analysis"
    if _has_compare_countries_task(state):
        return "comparison_agent"
    return "recommendation"


def should_run_country_comparison(
    state: GraphState,
) -> Literal["comparison_agent", "recommendation"]:
    """
    Decide whether a dedicated country comparison step is needed.

    Only used when the plan explicitly contains a compare_countries task.
    """
    plan = state.get("execution_plan", {})
    tasks = plan.get("tasks", [])
    has_comparison = any(t.get("task_type") == "compare_countries" for t in tasks)
    if has_comparison:
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

    # skill_extraction → trend_analysis OR resume_analysis (conditional)
    graph.add_conditional_edges(
        "skill_extraction",
        should_analyze_trends,
        {
            "trend_analysis": "trend_analysis",
            "resume_analysis": "resume_analysis",
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
