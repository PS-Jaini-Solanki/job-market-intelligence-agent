"""Regression tests for LangGraph conditional routing (agents/graph.py)."""

import pytest

from agents.graph import (
    build_graph,
    get_graph,
    should_analyze_resume,
    _has_compare_countries_task,
)
from agents.state import create_initial_state


def _state_with_plan(tasks):
    state = create_initial_state(user_query="test")
    state["execution_plan"] = {"tasks": tasks}
    return state


def test_has_compare_countries_task_true():
    state = _state_with_plan(
        [{"task_type": "collect_jobs"}, {"task_type": "compare_countries"}]
    )
    assert _has_compare_countries_task(state) is True


def test_has_compare_countries_task_false():
    state = _state_with_plan([{"task_type": "collect_jobs"}])
    assert _has_compare_countries_task(state) is False


def test_should_analyze_resume_with_resume_path():
    state = _state_with_plan([{"task_type": "collect_jobs"}])
    state["resume_path"] = "uploads/resumes/test.txt"
    state["query_intent"] = "resume_analysis"
    assert should_analyze_resume(state) == "resume_analysis"


def test_should_analyze_resume_runs_for_job_market_intent_too():
    # A resume + a job_market question ("here's my resume, what skills do
    # I need for X role?") is a legitimate combined question.
    state = _state_with_plan([{"task_type": "collect_jobs"}])
    state["resume_path"] = "uploads/resumes/test.txt"
    state["query_intent"] = "job_market"
    assert should_analyze_resume(state) == "resume_analysis"


def test_should_analyze_resume_skips_stale_resume_for_unrelated_intent():
    # Regression: a resume uploaded earlier in the conversation stays in
    # session state (chainlit_ui/app.py never clears it). Without this
    # intent check, asking a completely unrelated later question — e.g.
    # a salary question — would still force a resume gap analysis to run,
    # and (before final_answer.py was also fixed) silently override the
    # actual answer with a stale resume-fit report.
    state = _state_with_plan([{"task_type": "collect_jobs"}])
    state["resume_path"] = "uploads/resumes/test.txt"
    state["query_intent"] = "salary"
    assert should_analyze_resume(state) == "recommendation"


def test_should_analyze_resume_without_resume_routes_to_comparison():
    # Regression: without a resume_path, a pending compare_countries task
    # must still route to comparison_agent instead of being skipped.
    state = _state_with_plan(
        [{"task_type": "collect_jobs"}, {"task_type": "compare_countries"}]
    )
    assert should_analyze_resume(state) == "comparison_agent"


def test_should_analyze_resume_without_resume_or_comparison():
    state = _state_with_plan([{"task_type": "collect_jobs"}])
    assert should_analyze_resume(state) == "recommendation"


def test_graph_includes_final_answer_node():
    graph = get_graph()
    nodes = set(graph.get_graph().nodes.keys())
    assert "final_answer" in nodes


def test_graph_wires_recommendation_through_final_answer():
    # recommendation → final_answer → report_generation (concise answer is
    # composed before the full report is persisted).
    graph = get_graph()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("recommendation", "final_answer") in edges
    assert ("final_answer", "report_generation") in edges
    assert ("report_generation", "__end__") in edges
