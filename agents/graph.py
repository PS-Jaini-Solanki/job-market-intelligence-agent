"""
agents/graph.py
================
LangGraph execution graph — wires all agents together.

HOW LANGGRAPH WORKS:
1. You define a StateGraph with a shared state type.
2. You add nodes (agents) to the graph.
3. You add edges (connections) between nodes.
4. LangGraph executes the graph, calling each node in order.
5. Each node can conditionally route to different next nodes.

GRAPH FLOW:
START → planner → market_collection → skill_extraction
       → trend_analysis (conditional) → resume_analysis (conditional)
       → recommendation → report_generation → END

The planner's output determines which optional agents run.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

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
# These functions determine which node to go to next based on state.
# LangGraph uses them to build conditional edges in the graph.


def should_analyze_trends(
    state: GraphState,
) -> Literal["trend_analysis", "resume_analysis"]:
    """
    Decide whether to run trend analysis based on the plan.

    Runs trend analysis if:
    - The plan includes an analyze_trends task, OR
    - Technologies to analyze are present in state.
    """
    plan = state.get("execution_plan", {})
    tasks = plan.get("tasks", [])
    technologies = state.get("technologies_to_analyze", [])

    has_trend_task = any(t.get("task_type") == "analyze_trends" for t in tasks)

    if has_trend_task or technologies:
        return "trend_analysis"
    return "resume_analysis"


def should_analyze_resume(
    state: GraphState,
) -> Literal["resume_analysis", "recommendation"]:
    """
    Decide whether to run resume analysis based on the plan.

    Runs resume analysis if a resume_path exists in state.
    """
    if state.get("resume_path"):
        return "resume_analysis"
    return "recommendation"


def should_stop_on_error(
    state: GraphState,
) -> Literal["market_collection", END]:
    """
    Stop execution if a critical error occurred in the planner.
    """
    if state.get("should_stop"):
        logger.warning("Graph stopping due to critical error flag")
        return END
    return "market_collection"


# ============================================================
# Graph construction
# ============================================================


def build_graph() -> StateGraph:
    """
    Build and compile the multi-agent LangGraph.

    The graph defines the execution flow for all agent interactions.

    Returns:
        A compiled LangGraph StateGraph ready for execution.

    Usage:
        graph = build_graph()
        result = await graph.ainvoke(initial_state)
    """
    # Create the state graph
    graph = StateGraph(GraphState)

    # --- Add all agent nodes ---
    # Each node is a function that takes state and returns a partial update
    graph.add_node("planner", run_planner)
    graph.add_node("market_collection", run_market_collection)
    graph.add_node("skill_extraction", run_skill_extraction)
    graph.add_node("trend_analysis", run_trend_analysis)
    graph.add_node("resume_analysis", run_resume_analysis)
    graph.add_node("recommendation", run_recommendation)
    graph.add_node("report_generation", run_report_generation)

    # --- Add edges ---

    # START → planner (always)
    graph.add_edge(START, "planner")

    # planner → market_collection (or END if critical error)
    graph.add_conditional_edges(
        "planner",
        should_stop_on_error,
        {"market_collection": "market_collection", END: END},
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

    # trend_analysis → resume_analysis OR recommendation (conditional)
    graph.add_conditional_edges(
        "trend_analysis",
        should_analyze_resume,
        {
            "resume_analysis": "resume_analysis",
            "recommendation": "recommendation",
        },
    )

    # resume_analysis → recommendation (always)
    graph.add_edge("resume_analysis", "recommendation")

    # recommendation → report_generation (always)
    graph.add_edge("recommendation", "report_generation")

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
