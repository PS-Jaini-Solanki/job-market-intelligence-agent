import pytest
from tools.salary_parser_tool import parse_salary_from_text, compute_salary_statistics
from models.job import JobPosting, SalaryRange

def test_parse_salary_from_text_usd():
    text = "We offer a competitive salary of $120k to $150k per year."
    res = parse_salary_from_text(text)
    assert res is not None
    assert res.min_value == 120000.0
    assert res.max_value == 150000.0
    assert res.currency == "USD"

def test_parse_salary_from_text_inr_lpa():
    text = "Package: 15-20 LPA"
    res = parse_salary_from_text(text)
    assert res is not None
    assert res.min_value == 1500000.0
    assert res.max_value == 2000000.0
    assert res.currency == "INR"

def test_compute_salary_statistics():
    postings = [
        JobPosting(title="A", company="C", location="L", country="US", source="adzuna", salary=SalaryRange(min_value=100000, max_value=200000, currency="USD")),
        JobPosting(title="B", company="C", location="L", country="US", source="adzuna", salary=SalaryRange(min_value=150000, max_value=250000, currency="USD")),
        JobPosting(title="C", company="C", location="L", country="US", source="adzuna", salary=None, description="$120k"),
    ]

    stats = compute_salary_statistics(postings, role="Test", country="US")
    assert stats.sample_count == 3
    assert stats.currency == "USD"
    # Midpoints: 150000, 200000, 120000. Min: 120000.
    assert stats.min_salary == 120000.0


def test_compute_salary_statistics_rejects_implausible_structured_value():
    # Regression: a real observed bug — an Adzuna posting's structured
    # salary_min field (bypasses parse_salary_from_text entirely) came
    # back as $250, clearly not a real annual salary, and it was reported
    # as the "minimum salary" in the aggregate stats. Implausible values
    # must be filtered out regardless of whether they came from structured
    # API fields or text parsing.
    postings = [
        JobPosting(title="A", company="C", location="L", country="US", source="adzuna", salary=SalaryRange(min_value=250, max_value=250, currency="USD")),
        JobPosting(title="B", company="C", location="L", country="US", source="adzuna", salary=SalaryRange(min_value=120000, max_value=150000, currency="USD")),
    ]
    stats = compute_salary_statistics(postings, role="Test", country="US")
    assert stats.sample_count == 1
    assert stats.min_salary == 135000.0  # only the plausible posting counted
