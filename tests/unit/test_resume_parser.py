import pytest
from tools.resume_parser_tool import estimate_experience_years

def test_estimate_experience_years():
    assert estimate_experience_years("I have 5 years of experience in Python.") == 5.0
    assert estimate_experience_years("3+ years experience with React") == 3.0
    assert estimate_experience_years("Experience of 10+ years") == 10.0
    assert estimate_experience_years("No experience explicitly mentioned") is None
