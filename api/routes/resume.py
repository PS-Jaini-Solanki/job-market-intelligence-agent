"""
api/routes/resume.py
=====================
Resume upload and analysis endpoints.

POST /api/resume/upload   — Upload a resume file
GET  /api/resume/{id}/analysis — Get gap analysis for a resume
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from database.connection import get_session
from database.repositories import ResumeRepository
from tools.resume_parser_tool import (
    SUPPORTED_RESUME_EXTENSIONS,
    estimate_experience_years,
    validate_resume_file,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
    display_name = file.filename or "resume"

    # Cheap checks before reading the whole body
    suffix = Path(display_name).suffix.lower()
    if suffix not in SUPPORTED_RESUME_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix or 'none'}. "
                   f"Allowed: {', '.join(sorted(SUPPORTED_RESUME_EXTENSIONS))}",
        )
    content = await file.read()
    if len(content) > settings.MAX_RESUME_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_RESUME_SIZE_MB}MB",
        )

    # Unique on-disk name so two "resume.pdf" uploads never overwrite each other
    upload_dir = Path(settings.RESUME_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", display_name)
    file_path = upload_dir / f"{uuid.uuid4()}_{safe_name}"
    file_path.write_bytes(content)

    # Same validation as the chat UI: readable text with resume structure
    check = validate_resume_file(file_path, display_name=display_name)
    if not check.ok:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=check.reason.replace("**", ""),
        )
    raw_text, skill_names = check.text, check.skills
    experience_years = estimate_experience_years(raw_text)
    logger.info("Resume file saved", extra={"path": str(file_path)})

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
