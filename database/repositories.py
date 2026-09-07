"""
database/repositories.py
========================
Repository pattern for database access.

WHY the repository pattern?
- Agents and tools should NOT write raw SQL or SQLAlchemy queries.
- Instead, they call repository methods like:
    await skill_repo.save_statistics(stats)
    await query_repo.get_by_id(query_id)
- This makes testing easier (mock the repository, not the database).
- This isolates database logic from business logic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    MarketQueryModel,
    ResumeModel,
    SkillStatisticModel,
    TrendSnapshotModel,
    UserModel,
)
from models.skill import SkillStatistic
from models.trend import TechTrend

logger = logging.getLogger(__name__)


class UserRepository:
    """Handles all database operations for users."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: str) -> Optional[UserModel]:
        """Fetch a user by their ID."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create_anonymous(self) -> UserModel:
        """
        Return the anonymous user record.

        Used for queries submitted without authentication.
        """
        anonymous_id = "00000000-0000-0000-0000-000000000001"
        user = await self.get_by_id(anonymous_id)
        if user is None:
            user = UserModel(
                id=anonymous_id,
                name="Anonymous",
                email="anonymous@jobmarket.dev",
            )
            self.session.add(user)
            await self.session.flush()
        return user


class QueryRepository:
    """Handles all database operations for market queries."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        query: str,
        query_type: str,
        user_id: Optional[str] = None,
    ) -> MarketQueryModel:
        """
        Create a new market query record.

        Args:
            query: The user's query text.
            query_type: Detected type (e.g., 'job_market', 'salary').
            user_id: Optional user ID (None for anonymous queries).

        Returns:
            The created MarketQueryModel with status='pending'.
        """
        record = MarketQueryModel(
            id=str(uuid4()),
            query=query,
            query_type=query_type,
            user_id=user_id,
            status="pending",
        )
        self.session.add(record)
        await self.session.flush()
        logger.info("Created market query", extra={"query_id": record.id})
        return record

    async def mark_running(self, query_id: str) -> None:
        """Mark a query as currently running."""
        await self.session.execute(
            update(MarketQueryModel)
            .where(MarketQueryModel.id == query_id)
            .values(status="running")
        )

    async def mark_completed(
        self,
        query_id: str,
        result: dict,
        execution_time: float,
    ) -> None:
        """
        Mark a query as completed and save its result.

        Args:
            query_id: The query to update.
            result: The full JSON result from the agent.
            execution_time: How many seconds the query took.
        """
        await self.session.execute(
            update(MarketQueryModel)
            .where(MarketQueryModel.id == query_id)
            .values(
                status="completed",
                result=result,
                execution_time=execution_time,
                completed_at=datetime.utcnow(),
            )
        )
        logger.info(
            "Query completed",
            extra={"query_id": query_id, "execution_time": execution_time},
        )

    async def mark_failed(self, query_id: str, error: str) -> None:
        """Mark a query as failed with an error message."""
        await self.session.execute(
            update(MarketQueryModel)
            .where(MarketQueryModel.id == query_id)
            .values(
                status="failed",
                result={"error": error},
                completed_at=datetime.utcnow(),
            )
        )

    async def get_by_id(self, query_id: str) -> Optional[MarketQueryModel]:
        """Fetch a query record by ID.

        A malformed (non-UUID) id is treated as "not found" rather than
        letting the WHERE clause reach the DB — Postgres raises an
        invalid-input-syntax DataError for a bad UUID literal, which would
        otherwise surface as an unhandled 500 instead of a 404.
        """
        try:
            UUID(query_id)
        except (ValueError, AttributeError, TypeError):
            return None
        result = await self.session.execute(
            select(MarketQueryModel).where(MarketQueryModel.id == query_id)
        )
        return result.scalar_one_or_none()


class SkillRepository:
    """Handles all database operations for skill statistics."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_statistics(
        self,
        statistics: list[SkillStatistic],
        source: str,
        country: Optional[str] = None,
    ) -> None:
        """
        Save or update skill statistics.

        If a record for (normalized_name, source, country) already exists,
        it's updated with the new counts. Otherwise a new record is created.

        Args:
            statistics: List of SkillStatistic objects to save.
            source: Data source identifier string.
            country: Optional ISO country code.
        """
        for stat in statistics:
            # Check if record already exists
            existing = await self.session.execute(
                select(SkillStatisticModel).where(
                    SkillStatisticModel.normalized_name == stat.skill.normalized_name,
                    SkillStatisticModel.source == source,
                    SkillStatisticModel.country == country,
                )
            )
            record = existing.scalar_one_or_none()

            if record:
                # Update existing record
                record.occurrence_count = stat.occurrence_count
                record.total_postings = stat.total_postings_analyzed
                record.collected_at = datetime.utcnow()
            else:
                # Create new record
                record = SkillStatisticModel(
                    id=str(uuid4()),
                    skill_name=stat.skill.raw_name,
                    normalized_name=stat.skill.normalized_name,
                    occurrence_count=stat.occurrence_count,
                    total_postings=stat.total_postings_analyzed,
                    source=source,
                    country=country,
                )
                self.session.add(record)

        await self.session.flush()
        logger.info(
            "Saved skill statistics",
            extra={"count": len(statistics), "source": source},
        )

    async def get_top_skills(
        self,
        country: Optional[str] = None,
        limit: int = 20,
    ) -> list[SkillStatisticModel]:
        """
        Get the most frequently occurring skills.

        Args:
            country: Filter by country code, or None for all countries.
            limit: Maximum number of skills to return.

        Returns:
            List of SkillStatisticModel ordered by occurrence_count DESC.
        """
        query = select(SkillStatisticModel).order_by(
            SkillStatisticModel.occurrence_count.desc()
        ).limit(limit)

        if country:
            query = query.where(SkillStatisticModel.country == country)

        result = await self.session.execute(query)
        return list(result.scalars().all())


class TrendRepository:
    """Handles all database operations for technology trends."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_trends(self, trends: list[TechTrend]) -> None:
        """
        Save a list of technology trends as a new snapshot.

        Args:
            trends: List of TechTrend objects to persist.
        """
        for trend in trends:
            record = TrendSnapshotModel(
                id=str(uuid4()),
                technology=trend.technology,
                github_stars=trend.github_stats.stars if trend.github_stats else None,
                github_forks=trend.github_stats.forks if trend.github_stats else None,
                github_contributors=(
                    trend.github_stats.contributors if trend.github_stats else None
                ),
                job_count=trend.job_count,
                popularity_score=trend.popularity_score,
                growth_rate=trend.growth_rate_30d,
                sources=trend.sources,
            )
            self.session.add(record)

        await self.session.flush()
        logger.info("Saved trend snapshots", extra={"count": len(trends)})

    async def get_latest_for_technology(
        self, technology: str
    ) -> Optional[TrendSnapshotModel]:
        """
        Get the most recent trend snapshot for a specific technology.

        Args:
            technology: Technology name (exact match).

        Returns:
            The most recent TrendSnapshotModel, or None if not found.
        """
        result = await self.session.execute(
            select(TrendSnapshotModel)
            .where(TrendSnapshotModel.technology == technology)
            .order_by(TrendSnapshotModel.collected_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


class ResumeRepository:
    """Handles all database operations for resumes."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(
        self,
        file_name: str,
        raw_text: str,
        extracted_skills: list[str],
        experience_years: Optional[float],
        user_id: Optional[str] = None,
    ) -> ResumeModel:
        """
        Save a parsed resume to the database.

        Args:
            file_name: Original filename.
            raw_text: Extracted text content.
            extracted_skills: List of normalized skill name strings.
            experience_years: Years of experience, if determinable.
            user_id: Optional user ID.

        Returns:
            The saved ResumeModel.
        """
        record = ResumeModel(
            id=str(uuid4()),
            user_id=user_id,
            file_name=file_name,
            raw_text=raw_text,
            extracted_skills=extracted_skills,
            experience_years=experience_years,
        )
        self.session.add(record)
        await self.session.flush()
        logger.info("Saved resume", extra={"resume_id": record.id})
        return record

    async def get_by_id(self, resume_id: str) -> Optional[ResumeModel]:
        """Fetch a resume by ID.

        A malformed (non-UUID) id is treated as "not found" — see the same
        guard in QueryRepository.get_by_id for why.
        """
        try:
            UUID(resume_id)
        except (ValueError, AttributeError, TypeError):
            return None
        result = await self.session.execute(
            select(ResumeModel).where(ResumeModel.id == resume_id)
        )
        return result.scalar_one_or_none()
