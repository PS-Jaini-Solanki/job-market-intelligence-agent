"""
database/models.py
==================
SQLAlchemy ORM models — the Python representation of database tables.

Each class here maps directly to a table in PostgreSQL.

Design notes:
- UUIDs are used as primary keys (more portable than serial integers)
- JSONB columns store flexible data (skills list, settings, etc.)
- All timestamps are in UTC (always store UTC, display in local time)
- Relationships use SQLAlchemy's relationship() for easy joins
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from database.connection import Base


class UserModel(Base):
    """
    Represents a registered user.

    Maps to the 'users' table.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    resumes: Mapped[list[ResumeModel]] = relationship(
        "ResumeModel", back_populates="user", cascade="all, delete-orphan"
    )
    queries: Mapped[list[MarketQueryModel]] = relationship(
        "MarketQueryModel", back_populates="user"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r}>"


class ResumeModel(Base):
    """
    Stores an uploaded and parsed resume.

    Maps to the 'resumes' table.
    extracted_skills is stored as JSONB (a list of skill name strings).
    """

    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Full extracted text from the resume file
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # List of normalized skill name strings, stored as JSON array
    extracted_skills: Mapped[list] = mapped_column(JSONB, default=list)
    experience_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped[UserModel | None] = relationship("UserModel", back_populates="resumes")

    def __repr__(self) -> str:
        return f"<Resume id={self.id!r} file={self.file_name!r}>"


class MarketQueryModel(Base):
    """
    Records every market intelligence query and its result.

    Maps to the 'market_queries' table.
    Storing queries lets us cache results and analyze usage patterns.
    """

    __tablename__ = "market_queries"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Full JSON result from the agent execution
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    execution_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped[UserModel | None] = relationship("UserModel", back_populates="queries")

    def __repr__(self) -> str:
        return f"<MarketQuery id={self.id!r} status={self.status!r}>"


class SkillStatisticModel(Base):
    """
    Aggregated skill occurrence counts from job market data.

    Maps to the 'skill_statistics' table.
    Each row is a unique (skill, source, country) combination.
    """

    __tablename__ = "skill_statistics"
    __table_args__ = (
        UniqueConstraint("normalized_name", "source", "country", name="uq_skill_source_country"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    skill_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=0)
    total_postings: Mapped[int] = mapped_column(Integer, default=0)
    # e.g. 'adzuna+arbeitnow'
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # ISO country code, e.g. 'IN', 'US', or None for global
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<SkillStat skill={self.normalized_name!r} count={self.occurrence_count}>"


class TrendSnapshotModel(Base):
    """
    Time-series trend data for technologies.

    Maps to the 'trend_snapshots' table.
    Taking snapshots over time lets us compute growth rates.
    """

    __tablename__ = "trend_snapshots"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    technology: Mapped[str] = mapped_column(String(255), nullable=False)
    github_stars: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    github_forks: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    github_contributors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    job_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    popularity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    growth_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    # JSON array of source strings
    sources: Mapped[list] = mapped_column(JSONB, default=list)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<TrendSnapshot tech={self.technology!r} score={self.popularity_score}>"
