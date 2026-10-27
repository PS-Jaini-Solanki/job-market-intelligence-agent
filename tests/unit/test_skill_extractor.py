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


def test_extract_skills_go_language_detected_standalone():
    # Regression: bare "Go" (not "golang") was never detected at all.
    text = "5 years of experience with Go and distributed systems"
    assert "Go" in extract_skills_from_text(text)


def test_extract_skills_r_excludes_r_and_d():
    # Regression: "R&D" and "HR" false-positived as the R language.
    text = "strong R&D background and HR coordination"
    assert "R" not in extract_skills_from_text(text)
    text2 = "We use R for statistical analysis and modeling"
    assert "R" in extract_skills_from_text(text2)


def test_extract_skills_r_excludes_hyphenated_terms():
    # "Mask R-CNN" is a vision model, not the R language.
    assert "R" not in extract_skills_from_text("Trained Mask R-CNN and Faster R-CNN models")
    assert "R" in extract_skills_from_text("Python, R and SQL")


def test_extract_skills_claude_excludes_person_names():
    # Regression: a real observed bug — "Claude" is a common first name,
    # and bare word-boundary matching counted a recruiter's or hiring
    # manager's name as a mention of the Anthropic Claude AI product,
    # inflating that skill's reported demand percentage.
    text = "Contact our recruiter Claude Martinez for more information."
    assert "Anthropic Claude" not in extract_skills_from_text(text)

    text2 = "We use Claude for our customer support chatbot."
    assert "Anthropic Claude" in extract_skills_from_text(text2)
