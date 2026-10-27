import pytest
from agents.classifier import (
    _extract_role_from_query,
    _extract_country_from_query,
    _extract_country_names,
    _extract_technologies_from_query,
    classify_query,
    is_analytical,
    QueryIntent,
)


def test_extract_role():
    assert _extract_role_from_query("AI Engineer in India") == "AI Engineer"
    assert _extract_role_from_query("looking for data scientist jobs") == "Data Scientist"
    # A non-job-market question must NOT default to a role anymore.
    assert _extract_role_from_query("what is the weather like?") is None
    assert _extract_role_from_query("tell me a joke") is None


def test_extract_country():
    assert _extract_country_from_query("jobs in India") == "IN"
    assert _extract_country_from_query("jobs in the US") == "US"
    assert _extract_country_from_query("remote jobs") is None


def test_extract_technologies():
    techs = _extract_technologies_from_query("should I learn LangGraph or CrewAI?")
    assert "LangGraph" in techs
    assert "CrewAI" in techs
    assert _extract_technologies_from_query("what is the weather like?") == []


def test_classify_off_topic_question():
    result = classify_query("what is the weather like in Bangalore?")
    assert result.intent == QueryIntent.GENERAL_CHAT
    assert result.country == "IN"  # Bangalore is still detected as India


def test_classify_greeting():
    result = classify_query("Hi!")
    assert result.intent == QueryIntent.GENERAL_CHAT


def test_classify_job_market_question():
    result = classify_query("What skills are required for AI Engineers in India?")
    assert result.intent == QueryIntent.JOB_MARKET
    assert result.role == "AI Engineer"
    assert result.country == "IN"


def test_classify_job_market_question_with_in_demand_phrasing():
    # Regression test: "skills ... in demand for <role>" was previously
    # misrouted to TECH_TREND because "demand for" matches the trend
    # vocabulary and "in demand" also matches the generic job-market
    # vocabulary — the OR condition let trend win even with a named role.
    # A named role must win over ambiguous trend phrasing.
    result = classify_query("What skills are in demand for AI Engineers in India?")
    assert result.intent == QueryIntent.JOB_MARKET
    assert result.role == "AI Engineer"
    assert result.country == "IN"


def test_classify_salary_question():
    result = classify_query("What is the average salary of a Data Scientist in the UK?")
    assert result.intent == QueryIntent.SALARY
    assert result.role == "Data Scientist"
    assert result.country == "GB"


def test_classify_trend_question():
    result = classify_query("Should I learn LangGraph or CrewAI?")
    assert result.intent == QueryIntent.TECH_TREND
    assert "LangGraph" in result.technologies


def test_classify_country_comparison():
    result = classify_query("Compare AI job markets in India vs USA")
    assert result.intent == QueryIntent.COUNTRY_COMPARISON
    assert "IN" in result.countries
    assert "US" in result.countries


def test_extract_country_names_preserves_mention_order():
    # Countries must be returned in the order the user mentioned them,
    # not in the internal map order.
    assert _extract_country_names(
        "Compare AI job markets in France vs Germany"
    ) == ["FR", "DE"]
    assert _extract_country_names(
        "Germany or Canada for data roles?"
    ) == ["DE", "CA"]


def test_extract_primary_country_is_first_mentioned():
    # The primary country (used for fallback searches) should be the
    # one the user mentioned first, e.g. France in "France vs Germany".
    assert _extract_country_from_query(
        "Compare AI job markets in France vs Germany"
    ) == "FR"


def test_classify_resume_question():
    result = classify_query("analyze my resume for an AI engineer role")
    assert result.intent == QueryIntent.RESUME_ANALYSIS


def test_classify_follow_up_inherits_analytical_context():
    # "am I missing anything?" has no job-market signal on its own, but after
    # a skills question it should stay analytical instead of going to chat.
    history = [
        {"role": "user", "content": "what skills am I lacking to be a devops engineer"},
        {"role": "assistant", "content": "Here is your gap analysis..."},
    ]
    result = classify_query("am I missing anything?", conversation_context=history)
    assert is_analytical(result.intent.value)
    assert result.role == "DevOps Engineer"


def test_classify_follow_up_inherits_role_for_salary():
    # "what about salaries in India?" carries no role — it must inherit it.
    history = [
        {"role": "user", "content": "what skills are required for data engineers"},
    ]
    result = classify_query("what about salaries in India?", conversation_context=history)
    assert result.intent == QueryIntent.SALARY
    assert result.role == "Data Engineer"
    assert result.country == "IN"


def test_classify_new_role_question_does_not_inherit_stale_technologies():
    # Regression: a brand-new standalone question that names its own role
    # ("Cloud Engineer") must NOT inherit a technology name left over from
    # an earlier, unrelated question ("Python") — that leak caused a
    # spurious Technology Trends section to appear in an otherwise-correct
    # job_market report.
    history = [
        {"role": "user", "content": "should I learn Python for AI roles?"},
    ]
    result = classify_query(
        "what technology we can learn to be a good cloud engineer",
        conversation_context=history,
    )
    assert result.intent == QueryIntent.JOB_MARKET
    assert result.role == "Cloud Engineer"
    assert result.technologies == []


def test_classify_ambiguous_tech_follow_up_still_inherits():
    # A genuine ambiguous follow-up with no role of its own should still
    # inherit the technology thread from the previous turn.
    history = [
        {"role": "user", "content": "should I learn LangGraph or CrewAI?"},
    ]
    result = classify_query("what about MCP too?", conversation_context=history)
    assert result.intent == QueryIntent.TECH_TREND
    assert "MCP" in result.technologies


def test_classify_greeting_never_inherits_context():
    history = [
        {"role": "user", "content": "what skills are required for data engineers"},
    ]
    result = classify_query("thanks!", conversation_context=history)
    assert result.intent == QueryIntent.GENERAL_CHAT


def test_planner_uses_role_resolved_by_classifier():
    # The classifier owns conversation context; the planner takes its result.
    import asyncio
    from agents.planner import run_planner
    from agents.state import create_initial_state

    state = create_initial_state("what about salaries?")
    state.update(query_intent="salary", role="DevOps Engineer", country_filter="IN")
    result = asyncio.run(run_planner(state))
    assert result["role"] == "DevOps Engineer"
    assert result["country_filter"] == "IN"
    assert result["role_is_default"] is False


def test_planner_does_not_inherit_stale_technologies_for_new_role_query():
    # Regression: agents/planner.py re-derives facts from conversation_history
    # independently of agents/classifier.py (a second, separate place the
    # same "stale technology" leak could happen). A brand-new question that
    # names its own role must not inherit a technology name left over from
    # an earlier, unrelated question — that leak caused a spurious Technology
    # Trends section (about a technology the user didn't ask about this
    # turn) to appear in an otherwise-correct job_market report.
    import asyncio
    from agents.planner import run_planner
    from agents.state import create_initial_state

    history = [
        {"role": "user", "content": "should I learn Python for AI roles?"},
    ]
    state = create_initial_state(
        "what technology we can learn to be a good cloud engineer",
        conversation_history=history,
    )
    state["query_intent"] = "job_market"
    result = asyncio.run(run_planner(state))
    assert result["role"] == "Cloud Engineer"
    assert result["technologies_to_analyze"] == []


def test_planner_never_reads_role_from_history_itself():
    # Regression: the planner used to re-derive facts from history on its
    # own, so an off-topic turn the classifier left role-less could still
    # pick up an earlier role.
    import asyncio
    from agents.planner import run_planner
    from agents.state import create_initial_state

    history = [{"role": "user", "content": "what skills are required for devops engineers"}]
    state = create_initial_state("which skills are in demand?", conversation_history=history)
    state["query_intent"] = "job_market"
    result = asyncio.run(run_planner(state))
    assert result["role_is_default"] is True
    assert result["role"] != "DevOps Engineer"


def test_recommendation_filters_skills_user_already_has():
    from agents.recommendation import _filter_known_skills
    recs = [
        "Learn AWS — required in 14% of postings",
        "Learn Azure — required in 5% of postings",
    ]
    filtered = _filter_known_skills(recs, ["AWS"])
    assert len(filtered) == 1
    assert "Azure" in filtered[0]


def test_planner_adds_tasks_the_llm_plan_missed():
    from agents.planner import _ensure_required_tasks
    from models.agent import AgentTask, PlannerOutput, TaskType

    plan = PlannerOutput(
        tasks=[AgentTask(task_type=TaskType.GENERATE_REPORT, parameters={})],
        reasoning="llm plan", query_type="country_comparison",
    )
    _ensure_required_tasks(plan, "country_comparison", "AI Engineer", "IN", ["IN", "US"], [], False)
    types = [t.task_type for t in plan.tasks]
    assert TaskType.COLLECT_JOBS in types
    assert TaskType.COMPARE_COUNTRIES in types
    compare = next(t for t in plan.tasks if t.task_type == TaskType.COMPARE_COUNTRIES)
    assert compare.parameters["country_codes"] == ["IN", "US"]
