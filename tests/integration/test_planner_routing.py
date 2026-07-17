import pytest
from agents.planner import _extract_role_from_query, _extract_country_from_query

def test_extract_role():
    assert _extract_role_from_query("AI Engineer in India") == "AI Engineer"
    assert _extract_role_from_query("looking for data scientist jobs") == "Data Scientist"
    assert _extract_role_from_query("what is the weather like?") == "AI Engineer" # default

def test_extract_country():
    assert _extract_country_from_query("jobs in India") == "IN"
    assert _extract_country_from_query("jobs in the US") == "US"
    assert _extract_country_from_query("remote jobs") is None
