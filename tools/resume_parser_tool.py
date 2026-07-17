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
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from models.resume import Resume
from tools.skill_extractor_tool import extract_skills_from_text
from services.normalization_service import normalize_skill
from models.skill import Skill, SkillCategory
from tools.skill_extractor_tool import _infer_skill_category

logger = logging.getLogger(__name__)


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

    reader = PdfReader(str(file_path))
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

    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".doc": extract_text_from_docx,  # Try docx parser for .doc too
        ".txt": extract_text_from_txt,
        ".text": extract_text_from_txt,
    }

    extractor = extractors.get(extension)
    if not extractor:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {list(extractors.keys())}"
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
