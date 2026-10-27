"""
Regression tests for the query classifier's edge cases.

Each case is a message a reviewer or real user typed that used to be
misrouted: off-topic questions after an analytical one inheriting its
role, greetings swallowing real questions, "us" read as the USA, and
countries the job sources don't cover being answered with US data.
"""

import pytest

from agents.classifier import QueryIntent, classify_query

AI_INDIA = [
    {"role": "user", "content": "What skills are required for AI Engineers in India?"},
    {"role": "assistant", "content": "..."},
]


@pytest.mark.parametrize(
    "query",
    [
        "what are the top 5 restaurants in ahmedabad",
        "What's the weather in Bangalore?",
        "what about the weather in Bangalore?",
        "tell me a joke",
        "Who won the cricket match yesterday?",
        "ok",
        "now tell me",
    ],
)
def test_off_topic_after_analytical_question_stays_conversational(query):
    result = classify_query(query, AI_INDIA)
    assert result.intent == QueryIntent.GENERAL_CHAT
    assert result.role is None  # never inherits "AI Engineer"


@pytest.mark.parametrize(
    "query,intent,role,countries",
    [
        ("what about salaries?", QueryIntent.SALARY, "AI Engineer", ["IN"]),
        ("what about Germany?", QueryIntent.JOB_MARKET, "AI Engineer", ["DE"]),
        ("and for data scientists?", QueryIntent.JOB_MARKET, "Data Scientist", ["IN"]),
        ("am I missing anything?", QueryIntent.JOB_MARKET, "AI Engineer", ["IN"]),
        ("compare with the US", QueryIntent.COUNTRY_COMPARISON, "AI Engineer", ["IN", "US"]),
    ],
)
def test_elliptical_follow_ups_inherit_context(query, intent, role, countries):
    result = classify_query(query, AI_INDIA)
    assert result.intent == intent
    assert result.role == role
    assert result.countries == countries


def test_follow_up_chain_keeps_thread():
    history = AI_INDIA + [
        {"role": "user", "content": "what about salaries?"},
        {"role": "assistant", "content": "..."},
    ]
    result = classify_query("and in the UK?", history)
    assert result.intent == QueryIntent.SALARY
    assert result.role == "AI Engineer"
    assert result.countries == ["GB"]


@pytest.mark.parametrize(
    "query,intent",
    [
        ("Hi, what are the top skills for Python developers in Germany?", QueryIntent.JOB_MARKET),
        ("hello can you tell me data scientist salary in UK", QueryIntent.SALARY),
    ],
)
def test_greeting_does_not_swallow_a_real_question(query, intent):
    assert classify_query(query).intent == intent


def test_thanks_followed_by_question_answers_the_question():
    assert classify_query("thanks! what about salaries?", AI_INDIA).intent == QueryIntent.SALARY


@pytest.mark.parametrize("query", ["hi", "hello there", "thanks", "thank you so much!"])
def test_pure_pleasantries_are_chat(query):
    assert classify_query(query, AI_INDIA).intent == QueryIntent.GENERAL_CHAT


def test_pronoun_us_is_not_the_usa():
    assert classify_query("Can you tell us the top skills for data scientists").countries == []
    assert classify_query("jobs in the US").countries == ["US"]
    assert classify_query("SALARY OF ML ENGINEER IN US").countries == ["US"]


def test_comparison_is_not_paris():
    # "comparison" contains "paris"; substring matching used to add France.
    result = classify_query("salary comparison between India and US for data scientists")
    assert result.countries == ["IN", "US"]


def test_go_only_counts_as_a_technology_when_written_as_a_name():
    assert "Go" not in classify_query("Should I go for data science?").technologies
    assert "Go" in classify_query("Should I learn Go?").technologies


def test_pay_off_a_loan_is_not_a_salary_question():
    assert classify_query("I want to pay off my loan, any tips?", AI_INDIA).intent == QueryIntent.GENERAL_CHAT


@pytest.mark.parametrize(
    "query,unsupported",
    [
        ("jobs in Brazil", ["Brazil"]),
        ("Top skills for blockchain developers in Japan", ["Japan"]),
    ],
)
def test_unsupported_countries_are_detected_not_replaced_by_us(query, unsupported):
    result = classify_query(query)
    assert result.unsupported_countries == unsupported
    assert result.countries == []


def test_new_jersey_is_not_the_country_jersey():
    result = classify_query("data engineer jobs in New Jersey")
    assert result.countries == ["US"]
    assert result.unsupported_countries == []


def test_unsupported_follow_up_does_not_keep_old_country():
    result = classify_query("what about Japan?", AI_INDIA)
    assert result.unsupported_countries == ["Japan"]
    assert result.countries == []


def test_any_named_developer_role_is_used_as_is():
    assert classify_query("Top skills for blockchain developers").role == "Blockchain Developer"
    assert classify_query("what skills do I need to be a react developer").role == "React Developer"


def test_non_tech_profession_is_not_forced_into_a_tech_role():
    result = classify_query("what are the skills needed for a doctor in India")
    assert result.intent == QueryIntent.GENERAL_CHAT


def test_technology_comparison_is_a_trend_question():
    result = classify_query("Compare Python vs Rust")
    assert result.intent == QueryIntent.TECH_TREND
    assert set(result.technologies) == {"Python", "Rust"}


def test_attached_resume_with_vague_text_is_resume_analysis_in_context():
    result = classify_query("now tell me", AI_INDIA, resume_attached=True)
    assert result.intent == QueryIntent.RESUME_ANALYSIS
    assert result.role == "AI Engineer"
    assert result.country == "IN"


def test_attached_resume_does_not_override_an_analytical_question():
    result = classify_query("What is the average salary of a Data Scientist in the UK?",
                            resume_attached=True)
    assert result.intent == QueryIntent.SALARY
