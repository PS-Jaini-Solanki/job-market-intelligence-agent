"""Unit tests for the Final Answer agent (agents/final_answer.py)."""

from agents.final_answer import compose_final_answer
from agents.state import create_initial_state


def _state(**updates):
    state = create_initial_state(user_query="test query")
    state.update(updates)
    return state


SAMPLE_SKILL_STATS = [
    {"skill_name": "CI/CD", "occurrence_count": 18, "total_postings": 100, "percentage": 18.0, "sources": ["adzuna"]},
    {"skill_name": "Python", "occurrence_count": 15, "total_postings": 100, "percentage": 15.0, "sources": ["adzuna"]},
    {"skill_name": "AWS", "occurrence_count": 14, "total_postings": 100, "percentage": 14.0, "sources": ["adzuna"]},
]


def test_resume_answer_lists_missing_skills():
    state = _state(
        query_intent="resume_analysis",
        role="DevOps Engineer",
        country_filter="US",
        skill_statistics=SAMPLE_SKILL_STATS,
        resume_analysis={
            "gap_score": 60.0,
            "total_postings_analyzed": 100,
            "query_context": "DevOps Engineer in US",
            "sources": ["adzuna"],
            "matching_skills": [{"normalized_name": "CI/CD"}, {"normalized_name": "Python"}],
            "missing_skills": [
                {
                    "skill": {"normalized_name": "AWS"},
                    "market_demand_percentage": 14.0,
                    "priority": "MEDIUM",
                }
            ],
        },
        recommendations=["Learn AWS — required in 14% of postings"],
    )
    answer = compose_final_answer(state)

    assert "Market Fit Score: 60.0/100" in answer
    assert "AWS" in answer  # the missing skill is named explicitly
    assert "CI/CD" in answer  # a matching skill is shown
    assert "DevOps Engineer in US" in answer
    assert "Learn AWS" in answer  # recommendation is surfaced


def test_resume_answer_no_gaps_says_coverage_good():
    state = _state(
        query_intent="resume_analysis",
        role="DevOps Engineer",
        resume_analysis={
            "gap_score": 90.0,
            "total_postings_analyzed": 100,
            "query_context": "DevOps Engineer",
            "sources": ["adzuna"],
            "matching_skills": [{"normalized_name": "AWS"}, {"normalized_name": "Docker"}],
            "missing_skills": [],
        },
    )
    answer = compose_final_answer(state)
    assert "great coverage" in answer.lower()


def test_job_market_answer_shows_top_skills():
    state = _state(
        query_intent="job_market",
        role="AI Engineer",
        country_filter="IN",
        skill_statistics=SAMPLE_SKILL_STATS,
    )
    answer = compose_final_answer(state)
    assert "AI Engineer" in answer
    assert "CI/CD" in answer
    assert "18" in answer  # demand number present


def test_salary_answer_shows_numbers():
    state = _state(
        query_intent="salary",
        role="Data Scientist",
        country_filter="GB",
        salary_data={
            "min_salary": 65000.0,
            "max_salary": 120000.0,
            "median_salary": 90000.0,
            "currency": "GBP",
            "sample_count": 50,
        },
    )
    answer = compose_final_answer(state)
    assert "Data Scientist" in answer
    assert "GBP 90,000" in answer
    assert "50 live postings" in answer


def test_trend_answer_ranks_technologies():
    state = _state(
        query_intent="tech_trend",
        trend_data={
            "trends": [
                {"technology": "LangGraph", "popularity_score": 82.0, "job_demand_percentage": 4.0},
                {"technology": "MCP", "popularity_score": 74.0, "job_demand_percentage": 3.0},
            ]
        },
    )
    answer = compose_final_answer(state)
    assert "LangGraph" in answer
    assert "82.0" in answer
    assert "MCP" in answer


def test_comparison_answer_table():
    state = _state(
        query_intent="country_comparison",
        role="AI Engineer",
        country_comparison={
            "countries": {
                "IN": {
                    "posting_count": 120,
                    "salary": {"median": 3000000, "currency": "INR"},
                    "top_skills": [{"skill": "Python"}, {"skill": "PyTorch"}],
                },
                "US": {
                    "posting_count": 80,
                    "salary": {"median": 150000, "currency": "USD"},
                    "top_skills": [{"skill": "Python"}, {"skill": "AWS"}],
                },
            },
            "cross_country_insights": {"common_skills": ["Python"]},
        },
    )
    answer = compose_final_answer(state)
    assert "AI Engineer job market comparison" in answer
    assert "INR 3,000,000" in answer
    assert "USD 150,000" in answer


def test_stale_resume_analysis_does_not_override_unrelated_salary_answer():
    # Regression: a resume analyzed on an earlier conversation turn stays
    # in state for the rest of the session. Asking a completely unrelated
    # salary question afterward must answer the salary question, not
    # silently return the old resume-fit report.
    state = _state(
        query_intent="salary",
        role="Backend Engineer",
        country_filter="DE",
        salary_data={
            "min_salary": 50000.0,
            "max_salary": 80000.0,
            "median_salary": 65000.0,
            "currency": "EUR",
            "sample_count": 12,
        },
        resume_analysis={
            "gap_score": 62.5,
            "total_postings_analyzed": 40,
            "query_context": "AI Engineer in IN",
            "sources": ["adzuna"],
            "matching_skills": [{"normalized_name": "Python"}],
            "missing_skills": [],
        },
    )
    answer = compose_final_answer(state)
    assert "EUR 65,000" in answer
    assert "Market Fit Score" not in answer


def test_resume_analysis_still_wins_for_resume_intent_even_with_other_data():
    state = _state(
        query_intent="resume_analysis",
        role="AI Engineer",
        country_filter="IN",
        salary_data={"sample_count": 0},
        resume_analysis={
            "gap_score": 62.5,
            "total_postings_analyzed": 40,
            "query_context": "AI Engineer in IN",
            "sources": ["adzuna"],
            "matching_skills": [{"normalized_name": "Python"}],
            "missing_skills": [],
        },
    )
    answer = compose_final_answer(state)
    assert "Market Fit Score" in answer


def test_no_data_fallback():
    state = _state(query_intent="job_market", role="AI Engineer")
    answer = compose_final_answer(state)
    assert "couldn't find enough live postings" in answer
