"""Tests for tools/resume_parser_tool.validate_resume_file (upload gatekeeping)."""

from tools.resume_parser_tool import validate_resume_file

RESUME_TEXT = """Jane Doe
jane.doe@example.com | +91 98765 43210 | linkedin.com/in/janedoe

Summary
Backend engineer with 4 years of experience building APIs.

Experience
Software Engineer, Acme Corp (2021-2025): Python, FastAPI, PostgreSQL, Docker, AWS.

Education
B.Tech in Computer Science

Skills
Python, Django, FastAPI, PostgreSQL, Docker, Kubernetes, AWS
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_real_resume_is_accepted_with_skills(tmp_path):
    result = validate_resume_file(_write(tmp_path, "resume.txt", RESUME_TEXT))
    assert result.ok
    assert {"Python", "Docker", "AWS"} <= set(result.skills)


def test_csv_is_rejected_by_format(tmp_path):
    result = validate_resume_file(_write(tmp_path, "coverage.csv", "id,field,coverage\n1,a,90\n"))
    assert not result.ok
    assert "isn't a supported resume format" in result.reason


def test_legacy_doc_is_rejected_because_it_cannot_be_parsed(tmp_path):
    assert not validate_resume_file(_write(tmp_path, "resume.doc", RESUME_TEXT)).ok


def test_text_file_that_is_not_a_resume_is_rejected(tmp_path):
    report = "Quarterly field coverage report. " * 20
    result = validate_resume_file(_write(tmp_path, "report.txt", report))
    assert not result.ok
    assert "doesn't look like a resume" in result.reason


def test_empty_file_is_rejected(tmp_path):
    result = validate_resume_file(_write(tmp_path, "resume.txt", "   "))
    assert not result.ok
    assert "readable text" in result.reason


def test_size_limit_is_enforced(tmp_path):
    path = _write(tmp_path, "resume.txt", RESUME_TEXT * 2000)
    result = validate_resume_file(path, max_size_mb=0.01)
    assert not result.ok
    assert "MB limit" in result.reason


def test_uploaded_name_decides_format_when_temp_path_has_no_extension(tmp_path):
    path = _write(tmp_path, "upload_tmp", RESUME_TEXT)
    assert validate_resume_file(path, display_name="resume.txt").ok
    assert not validate_resume_file(path, display_name="data.csv").ok
