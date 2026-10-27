"""
config.py
=========
Application configuration using pydantic-settings.

pydantic-settings automatically reads values from:
1. Environment variables (highest priority)
2. .env file (if present)
3. Default values defined in the model

This means you NEVER hardcode secrets in the code.
You put them in .env and pydantic-settings loads them automatically.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings, loaded from environment variables / .env file.

    Usage:
        from config import get_settings
        settings = get_settings()
        print(settings.GROQ_API_KEY)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore unknown env variables
    )

    # ----------------------------------------------------------
    # LLM Settings
    # ----------------------------------------------------------
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    GROQ_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model to use",
    )
    GROQ_MAX_TOKENS: int = Field(default=4096)
    GROQ_TEMPERATURE: float = Field(default=0.1)

    # ----------------------------------------------------------
    # External API Keys
    # ----------------------------------------------------------
    GITHUB_TOKEN: str = Field(default="", description="GitHub personal access token")
    TAVILY_API_KEY: str = Field(default="", description="Tavily search API key")
    ADZUNA_APP_ID: str = Field(default="", description="Adzuna API app ID")
    ADZUNA_APP_KEY: str = Field(default="", description="Adzuna API app key")

    # Optional Reddit
    REDDIT_CLIENT_ID: Optional[str] = Field(default=None)
    REDDIT_CLIENT_SECRET: Optional[str] = Field(default=None)
    REDDIT_USER_AGENT: str = Field(default="JobMarketIntelligenceBot/1.0")

    # ----------------------------------------------------------
    # Database
    # ----------------------------------------------------------
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="job_market_db")
    POSTGRES_USER: str = Field(default="job_market_user")
    POSTGRES_PASSWORD: str = Field(default="changeme")
    SQLALCHEMY_DATABASE_URL: str = Field(
        default="postgresql+asyncpg://job_market_user:changeme@localhost:5432/job_market_db"
    )

    # ----------------------------------------------------------
    # FastAPI
    # ----------------------------------------------------------
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_DEBUG: bool = Field(default=True)
    SECRET_KEY: str = Field(default="change-this-in-production")

    # ----------------------------------------------------------
    # Chainlit
    # ----------------------------------------------------------
    CHAINLIT_HOST: str = Field(default="0.0.0.0")
    CHAINLIT_PORT: int = Field(default=8001)

    # ----------------------------------------------------------
    # Agent Settings
    # ----------------------------------------------------------
    MAX_JOB_POSTINGS: int = Field(default=100)
    MAX_RETRIES: int = Field(default=3)
    API_TIMEOUT: int = Field(default=30)

    # ----------------------------------------------------------
    # File Upload
    # ----------------------------------------------------------
    RESUME_UPLOAD_DIR: str = Field(default="./uploads/resumes")
    MAX_RESUME_SIZE_MB: int = Field(default=10)

    # ----------------------------------------------------------
    # Computed properties (not loaded from env)
    # ----------------------------------------------------------
    @computed_field
    @property
    def database_url_masked(self) -> str:
        """Return database URL with password masked (safe to log)."""
        return self.SQLALCHEMY_DATABASE_URL.replace(self.POSTGRES_PASSWORD, "***")

    # Placeholder values shipped in .env.example — treat as "not configured"
    # so a user who never filled these in doesn't get silent, confusing
    # failures from reddit_mcp trying to authenticate with fake credentials.
    _PLACEHOLDER_VALUES: set[str] = {
        "your_reddit_client_id",
        "your_reddit_client_secret",
    }

    @computed_field
    @property
    def has_reddit(self) -> bool:
        """True if real (non-placeholder) Reddit credentials are configured."""
        client_id = (self.REDDIT_CLIENT_ID or "").strip()
        client_secret = (self.REDDIT_CLIENT_SECRET or "").strip()
        if client_id in self._PLACEHOLDER_VALUES or client_secret in self._PLACEHOLDER_VALUES:
            return False
        return bool(client_id and client_secret)

    @computed_field
    @property
    def has_adzuna(self) -> bool:
        """True if Adzuna credentials are configured."""
        return bool(self.ADZUNA_APP_ID and self.ADZUNA_APP_KEY)

    @computed_field
    @property
    def has_tavily(self) -> bool:
        """True if Tavily credentials are configured."""
        return bool(self.TAVILY_API_KEY)

    @computed_field
    @property
    def has_github(self) -> bool:
        """True if GitHub token is configured."""
        return bool(self.GITHUB_TOKEN)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings (cached after first call).

    Using lru_cache means the .env file is only read once.
    The cache is cleared in tests using get_settings.cache_clear().
    """
    return Settings()
