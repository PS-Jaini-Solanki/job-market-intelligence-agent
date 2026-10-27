"""
tools/resume_parser_tool.py
============================
Extracts text content from uploaded resume files.

SUPPORTED FORMATS:
- PDF (.pdf) — most common
- Word Document (.docx)
- Plain text (.txt)

HOW IT WORKS:
1. Detect the file type from the extension.
2. Use the appropriate library to extract raw text.
3. Clean up whitespace and encoding artifacts.
4. Return the raw text for further processing by the skill extractor.

We intentionally return raw text rather than pre-extracted skills.
This separation of concerns means:
- The parser is testable without the skill extractor.
- We can re-run skill extraction without re-parsing the file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from models.resume import Resume
from tools.skill_extractor_tool import extract_skills_from_text
from services.normalization_service import normalize_skill
from models.skill import Skill, SkillCategory
from tools.skill_extractor_tool import _infer_skill_category

logger = logging.getLogger(__name__)

# Below this much text a "resume" is empty or a scanned image.
MIN_RESUME_TEXT_CHARS = 150

# Headings that resumes have and arbitrary documents (CSV exports,
# invoices, reports) don't.
_RESUME_SECTION_RE = re.compile(
    r"\b(experience|work history|employment|education|skills|projects?|internships?"
    r"|certifications?|summary|objective|achievements|qualifications|profile)\b",
    re.IGNORECASE,
)
_CONTACT_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|linkedin\.com|github\.com|\+?\d[\d\s().-]{8,}\d",
    re.IGNORECASE,
)


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text string.

    Raises:
        ImportError: If pypdf is not installed.
        ValueError: If the PDF cannot be read.
    """
    try:
        # pyrefly: ignore [missing-import]
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf is required for PDF parsing: pip install pypdf") from e

    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        # PdfReader construction itself can raise (empty file, corrupted
        # stream, encrypted PDF, etc.) — this was previously uncaught,
        # so a bad upload crashed with a raw pypdf exception instead of
        # the ValueError this function's docstring promises.
        raise ValueError(f"Corrupted or unreadable PDF file: {e}") from e

    text_parts = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        except Exception as e:
            logger.warning(
                "Failed to extract text from PDF page",
                extra={"page": page_num, "error": str(e)},
            )

    return "\n".join(text_parts)


def extract_text_from_docx(file_path: Path) -> str:
    """
    Extract text from a Word (.docx) file.

    Args:
        file_path: Path to the .docx file.

    Returns:
        Extracted text string.
    """
    try:
        # pyrefly: ignore [missing-import]
        from docx import Document
    except ImportError as e:
        raise ImportError("python-docx is required: pip install python-docx") from e

    doc = Document(str(file_path))
    text_parts = []

    # Extract text from paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)

    # Also extract text from tables (skills are often in tables)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text)

    return "\n".join(text_parts)


def extract_text_from_txt(file_path: Path) -> str:
    """
    Extract text from a plain text file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        File contents as a string.
    """
    # Try UTF-8 first, fall back to latin-1
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot decode text file: {file_path}")


# The formats we can actually extract text from. Legacy binary .doc is not
# here: python-docx only reads .docx, so accepting .doc would fail silently.
_EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
}
SUPPORTED_RESUME_EXTENSIONS: frozenset[str] = frozenset(_EXTRACTORS)


def extract_text_from_file(file_path: str | Path) -> str:
    """
    Extract raw text from a resume file, detecting format automatically.

    Args:
        file_path: Path to the resume file (PDF, DOCX, or TXT).

    Returns:
        Extracted text as a string.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    extension = path.suffix.lower()

    extractor = _EXTRACTORS.get(extension)
    if not extractor:
        raise ValueError(
            f"Unsupported file format: {extension or 'no extension'}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_RESUME_EXTENSIONS))}"
        )

    raw_text = extractor(path)
    return _clean_extracted_text(raw_text)


def _clean_extracted_text(text: str) -> str:
    """
    Clean up common artifacts from PDF/DOCX text extraction.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    # Replace multiple whitespace/newlines with single space or newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Remove null bytes and other control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def estimate_experience_years(text: str) -> Optional[float]:
    """
    Attempt to estimate years of experience from resume text.

    This looks for patterns like "5+ years", "3-5 years experience", etc.
    Returns None if the pattern cannot be reliably detected.

    Args:
        text: Resume text to search.

    Returns:
        Estimated years of experience, or None.
    """
    # Look for explicit experience statements
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
        r"(\d+)\+?\s*yrs?\s*(?:of\s*)?experience",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


@dataclass
class ResumeValidation:
    """Outcome of checking an uploaded file before treating it as a resume."""

    ok: bool
    reason: str = ""
    text: str = ""
    skills: list[str] = field(default_factory=list)


def validate_resume_file(
    file_path: str | Path,
    display_name: Optional[str] = None,
    max_size_mb: Optional[float] = None,
) -> ResumeValidation:
    """
    Decide whether an uploaded file is a resume we can analyze.

    Checks, in order: supported format, size limit, extractable text, and
    resume structure (resume section headings, contact details, or several
    technical skills). Anything else (a CSV export, an invoice, a scanned
    image) is rejected with a reason the user can act on, instead of being
    scored as a resume with zero skills.

    Args:
        file_path: Where the uploaded file is on disk.
        display_name: The name the user uploaded it as (defaults to the path name).
        max_size_mb: Size limit; None means no limit.

    Returns:
        ResumeValidation with ok, a user-facing reason, and the parsed text/skills.
    """
    path = Path(file_path)
    name = display_name or path.name
    extension = Path(name).suffix.lower() or path.suffix.lower()

    if extension not in SUPPORTED_RESUME_EXTENSIONS:
        return ResumeValidation(
            ok=False,
            reason=(
                f"**{name}** isn't a supported resume format. Please upload a "
                f"{', '.join(sorted(SUPPORTED_RESUME_EXTENSIONS))} file."
            ),
        )

    if max_size_mb is not None and path.stat().st_size > max_size_mb * 1024 * 1024:
        return ResumeValidation(
            ok=False, reason=f"**{name}** is larger than the {max_size_mb:g} MB limit."
        )

    try:
        # Parse by the uploaded name's extension: temp upload paths may have none.
        text = _clean_extracted_text(_EXTRACTORS[extension](path))
    except Exception as e:
        logger.warning("Resume text extraction failed", extra={"file": name, "error": str(e)})
        return ResumeValidation(
            ok=False, reason=f"I couldn't read **{name}**; the file may be corrupted or protected."
        )

    if len(text) < MIN_RESUME_TEXT_CHARS:
        return ResumeValidation(
            ok=False,
            reason=(
                f"I couldn't find readable text in **{name}**. If it's a scanned "
                "image, please upload a text-based PDF, DOCX, or TXT instead."
            ),
        )

    skills = extract_skills_from_text(text)
    sections = {m.lower() for m in _RESUME_SECTION_RE.findall(text)}
    has_contact = bool(_CONTACT_RE.search(text))
    looks_like_resume = (
        len(sections) >= 2
        or (sections and has_contact)
        or (len(skills) >= 3 and has_contact)
    )
    if not looks_like_resume:
        return ResumeValidation(
            ok=False,
            reason=(
                f"**{name}** doesn't look like a resume (no sections like Experience, "
                "Education or Skills, and no contact details). Please upload your resume."
            ),
        )

    return ResumeValidation(ok=True, text=text, skills=skills)


@tool
def parse_resume(file_path: str) -> dict:
    """
    Parse a resume file and extract its text content and skills.

    Supports PDF, DOCX, and TXT formats.
    Skills are extracted using keyword matching (not LLM generation).

    Args:
        file_path: Absolute path to the resume file.

    Returns:
        Serialized Resume object with extracted text and skills.
    """
    logger.info("Parsing resume", extra={"file_path": file_path})

    path = Path(file_path)

    # Extract raw text
    raw_text = extract_text_from_file(path)

    # Extract skills from the text
    skill_names = extract_skills_from_text(raw_text)
    skills = [
        Skill(
            raw_name=name,
            normalized_name=name,
            category=_infer_skill_category(name),
        )
        for name in skill_names
    ]

    # Estimate experience
    experience_years = estimate_experience_years(raw_text)

    resume = Resume(
        file_name=path.name,
        raw_text=raw_text,
        extracted_skills=skills,
        experience_years=experience_years,
    )

    logger.info(
        "Resume parsed",
        extra={
            "file": path.name,
            "skills_found": len(skills),
            "experience_years": experience_years,
        },
    )

    return resume.model_dump()
