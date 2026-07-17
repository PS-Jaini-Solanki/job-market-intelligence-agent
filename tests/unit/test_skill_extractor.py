import pytest
from tools.skill_extractor_tool import extract_skills_from_text

def test_extract_skills_from_text():
    text = "Looking for a Python developer with experience in Django and AWS. Should know React."
    skills = extract_skills_from_text(text)
    
    assert "Python" in skills
    assert "Django" in skills
    assert "AWS" in skills
    assert "React" in skills
    assert "Java" not in skills

def test_extract_skills_case_insensitive():
    text = "We use python, django, and aws."
    skills = extract_skills_from_text(text)
    
    assert "Python" in skills
    assert "Django" in skills
    assert "AWS" in skills

def test_extract_skills_word_boundaries():
    # 'go' shouldn't match 'going'
    # 'c' shouldn't match 'can'
    text = "Going to use React."
    skills = extract_skills_from_text(text)
    
    assert "Go" not in skills
    assert "C" not in skills
    assert "React" in skills
