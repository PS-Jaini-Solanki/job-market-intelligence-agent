"""Unit tests for the Report Generation agent (agents/report_generation.py)."""

import asyncio

from agents.report_generation import run_report_generation
from agents.state import create_initial_state


def test_no_report_when_resume_intent_has_no_resume_attached():
    # Regression: a real observed bug — a resume-analysis question with no
    # resume ever attached still produced a full, polished "Job Market
    # Intelligence Report" (using side-collected market data for a
    # defaulted role), directly contradicting the chat answer, which
    # correctly said no resume was found. No report should be generated
    # when there's nothing real to report on.
    state = create_initial_state("analyse the resume I have attached")
    state["query_intent"] = "resume_analysis"
    state["skill_statistics"] = [
        {"skill_name": "Python", "occurrence_count": 10, "total_postings": 50, "percentage": 20.0, "sources": ["adzuna"]}
    ]
    result = asyncio.run(run_report_generation(state))
    assert result["final_report"] is None


def test_report_still_generated_when_resume_actually_present():
    state = create_initial_state("analyse my resume", resume_path="uploads/resumes/fake.pdf")
    state["query_intent"] = "resume_analysis"
    state["resume_data"] = {"file_name": "fake.pdf", "extracted_skills": []}
    result = asyncio.run(run_report_generation(state))
    assert result["final_report"] is not None
