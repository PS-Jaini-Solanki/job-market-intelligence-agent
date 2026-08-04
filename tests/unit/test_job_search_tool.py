"""Regression tests for job_search_tool.py helpers."""

import pytest
from datetime import datetime

from tools.job_search_tool import (
    ADZUNA_COUNTRY_CURRENCIES,
    _parse_date,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-01-15T10:30:00Z", datetime(2024, 1, 15, 10, 30, 0)),
        ("2024-01-15", datetime(2024, 1, 15)),
        (1705314600, datetime.fromtimestamp(1705314600)),
        (1705314600.0, datetime.fromtimestamp(1705314600.0)),
    ],
)
def test_parse_date_valid(raw, expected):
    assert _parse_date(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "not-a-date", 0])
def test_parse_date_invalid_returns_none(raw):
    assert _parse_date(raw) is None


def test_adzuna_currency_per_country():
    # Adzuna reports salaries in the local currency of each country board,
    # so the labels must match — never default to USD for non-US markets.
    assert ADZUNA_COUNTRY_CURRENCIES["IN"] == "INR"
    assert ADZUNA_COUNTRY_CURRENCIES["US"] == "USD"
    assert ADZUNA_COUNTRY_CURRENCIES["GB"] == "GBP"
    assert ADZUNA_COUNTRY_CURRENCIES["AU"] == "AUD"
    assert ADZUNA_COUNTRY_CURRENCIES["CA"] == "CAD"
    assert ADZUNA_COUNTRY_CURRENCIES["DE"] == "EUR"
    assert ADZUNA_COUNTRY_CURRENCIES["FR"] == "EUR"
    assert ADZUNA_COUNTRY_CURRENCIES["SG"] == "SGD"
    assert ADZUNA_COUNTRY_CURRENCIES["AE"] == "AED"
    # Every supported country path has a labelled currency.
    for code in ADZUNA_COUNTRY_CURRENCIES:
        assert ADZUNA_COUNTRY_CURRENCIES[code]
