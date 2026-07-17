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
        JobPosting(title="A", company="C", location="L", source="adzuna", salary=SalaryRange(min_value=100, max_value=200, currency="USD")),
        JobPosting(title="B", company="C", location="L", source="adzuna", salary=SalaryRange(min_value=150, max_value=250, currency="USD")),
        JobPosting(title="C", company="C", location="L", source="adzuna", salary=None, description="$120k"),
    ]
    
    stats = compute_salary_statistics(postings, role="Test", country="US")
    assert stats.sample_count == 3
    assert stats.currency == "USD"
    # Midpoints: 150, 200, 120000. Min: 150.
    assert stats.min_salary == 150.0
