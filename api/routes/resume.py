"""
api/routes/resume.py
=====================
Resume upload and analysis endpoints.

POST /api/resume/upload   — Upload a resume file
GET  /api/resume/{id}/analysis — Get gap analysis for a resume
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from database.connection import get_session
from database.repositories import ResumeRepository

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


class ResumeUploadResponse(BaseModel):
    """Response from a resume upload."""
    resume_id: str
    file_name: str
    skills_found: int
    file_path: str
    message: str


class ResumeAnalysisResponse(BaseModel):
    """Response with resume gap analysis."""
    resume_id: str
    gap_score: float
    matching_skills: list[str]
    missing_skills_high_priority: list[str]
    missing_skills_medium_priority: list[str]
    experience_years: float | None
    summary: str


@router.post(
    "/resume/upload",
    response_model=ResumeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a resume file",
    description="Upload a PDF, DOCX, or TXT resume. Skills are extracted automatically.",
)
async def upload_resume(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    session: AsyncSession = Depends(get_session),
) -> ResumeUploadResponse:
    """
    Upload a resume and extract skills from it.

    The resume is saved to disk and parsed using the resume_parser_tool.
    Extracted skills are stored in the database.

    Args:
        file: The uploaded file.
        session: Database session.

    Returns:
        ResumeUploadResponse with the resume ID and extracted skill count.
    """
    settings = get_settings()

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Validate file size
    content = await file.read()
    max_bytes = settings.MAX_RESUME_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_RESUME_SIZE_MB}MB",
        )

    # Save file to disk
    upload_dir = Path(settings.RESUME_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename but sanitize it
    import re
    safe_name = re.sub(r"[^\w\-.]", "_", file.filename or "resume")
    file_path = upload_dir / safe_name

    file_path.write_bytes(content)
    logger.info("Resume file saved", extra={"path": str(file_path)})

    # Parse the resume
    try:
        from tools.resume_parser_tool import extract_text_from_file, estimate_experience_years
        from tools.skill_extractor_tool import extract_skills_from_text

        raw_text = extract_text_from_file(file_path)
        skill_names = extract_skills_from_text(raw_text)
        experience_years = estimate_experience_years(raw_text)

    except Exception as e:
        logger.error("Resume parsing failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse resume: {str(e)}",
        )

    # Save to database
    resume_repo = ResumeRepository(session)
    resume_record = await resume_repo.save(
        file_name=safe_name,
        raw_text=raw_text,
        extracted_skills=skill_names,
        experience_years=experience_years,
    )

    logger.info(
        "Resume processed",
        extra={"resume_id": resume_record.id, "skills": len(skill_names)},
    )

    return ResumeUploadResponse(
        resume_id=resume_record.id,
        file_name=safe_name,
        skills_found=len(skill_names),
        file_path=str(file_path),
        message=(
            f"Resume uploaded successfully. "
            f"Found {len(skill_names)} skills. "
            "Use the Chainlit UI to run a gap analysis."
        ),
    )


@router.get(
    "/resume/{resume_id}/skills",
    summary="Get skills extracted from a resume",
)
async def get_resume_skills(
    resume_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Get the skills extracted from a specific resume.

    Args:
        resume_id: The resume ID from the upload response.
        session: Database session.

    Returns:
        Dictionary with resume metadata and extracted skills.
    """
    resume_repo = ResumeRepository(session)
    record = await resume_repo.get_by_id(resume_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume '{resume_id}' not found",
        )

    return {
        "resume_id": record.id,
        "file_name": record.file_name,
        "extracted_skills": record.extracted_skills,
        "experience_years": record.experience_years,
        "uploaded_at": record.upload_timestamp.isoformat(),
    }
