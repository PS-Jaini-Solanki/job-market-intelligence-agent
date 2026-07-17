import pytest
from services.normalization_service import normalize_skill

def test_normalize_skill_exact_match():
    assert normalize_skill("Python") == "Python"
    assert normalize_skill("AWS") == "AWS"

def test_normalize_skill_alias():
    assert normalize_skill("Amazon Web Services") == "AWS"
    assert normalize_skill("google cloud") == "GCP"
    assert normalize_skill("react.js") == "React"

def test_normalize_skill_whitespace_and_case():
    assert normalize_skill("  python  ") == "Python"
    assert normalize_skill("aWs") == "AWS"
    assert normalize_skill("ReactJS") == "React"

def test_normalize_skill_unknown():
    assert normalize_skill("Some New Tool") == "Some New Tool"
