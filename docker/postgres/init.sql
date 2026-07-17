-- ============================================================
-- Job Market Intelligence Agent — Initial Database Schema
-- ============================================================
-- This SQL file is automatically executed by PostgreSQL when
-- the Docker container starts for the first time.
--
-- Note: Alembic migrations will manage schema changes after
-- the initial setup. This file is only for first-time setup.
-- ============================================================

-- Enable UUID generation extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Users table
-- Stores user accounts
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(255) NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- Resumes table
-- Stores uploaded resumes and their extracted skills
-- ============================================================
CREATE TABLE IF NOT EXISTS resumes (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID REFERENCES users(id) ON DELETE CASCADE,
    file_name           VARCHAR(255) NOT NULL,
    raw_text            TEXT,
    extracted_skills    JSONB DEFAULT '[]',   -- List of skill strings
    experience_years    FLOAT,
    upload_timestamp    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- Market queries table
-- Stores every query submitted by users and its result
-- ============================================================
CREATE TABLE IF NOT EXISTS market_queries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    query           TEXT NOT NULL,
    query_type      VARCHAR(100),  -- e.g. 'job_market', 'salary', 'comparison'
    result          JSONB,         -- Full JSON result
    execution_time  FLOAT,         -- Seconds taken to complete
    status          VARCHAR(50) DEFAULT 'pending', -- pending | running | done | failed
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE
);

-- ============================================================
-- Skill statistics table
-- Stores aggregated skill occurrence counts from job data
-- ============================================================
CREATE TABLE IF NOT EXISTS skill_statistics (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    skill_name      VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    occurrence_count INTEGER DEFAULT 0,
    total_postings  INTEGER DEFAULT 0,
    source          VARCHAR(255),  -- e.g. 'adzuna+arbeitnow'
    country         VARCHAR(100),  -- e.g. 'IN', 'US'
    collected_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(normalized_name, source, country)
);

-- ============================================================
-- Trend snapshots table
-- Stores GitHub + job market trend data over time
-- ============================================================
CREATE TABLE IF NOT EXISTS trend_snapshots (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    technology          VARCHAR(255) NOT NULL,
    github_stars        INTEGER,
    github_forks        INTEGER,
    github_contributors INTEGER,
    job_count           INTEGER,
    popularity_score    FLOAT,      -- Computed composite score
    growth_rate         FLOAT,      -- % change from previous snapshot
    sources             JSONB DEFAULT '[]',
    collected_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_market_queries_user_id     ON market_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_market_queries_created_at  ON market_queries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_skill_statistics_skill     ON skill_statistics(normalized_name);
CREATE INDEX IF NOT EXISTS idx_skill_statistics_country   ON skill_statistics(country);
CREATE INDEX IF NOT EXISTS idx_trend_snapshots_tech       ON trend_snapshots(technology);
CREATE INDEX IF NOT EXISTS idx_trend_snapshots_collected  ON trend_snapshots(collected_at DESC);

-- ============================================================
-- Seed data: default anonymous user for unauthenticated queries
-- ============================================================
INSERT INTO users (id, name, email)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Anonymous',
    'anonymous@jobmarket.dev'
)
ON CONFLICT (email) DO NOTHING;
