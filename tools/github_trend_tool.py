"""
tools/github_trend_tool.py
===========================
Fetches technology trend data from the GitHub API.

WHY GITHUB DATA?
GitHub activity is one of the best publicly available signals for
technology adoption:
- Stars: Community interest and adoption
- Forks: Active developer usage  
- Recent commits: Project health and activity
- Contributors: Community size
- Release frequency: Project maturity

HALLUCINATION PREVENTION:
All data comes directly from the GitHub API.
We never estimate or approximate GitHub statistics.

RATE LIMITS:
- Without a token: 60 requests/hour
- With a token: 5,000 requests/hour
We strongly recommend setting GITHUB_TOKEN in your .env file.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from models.trend import GitHubStats, TechTrend, TrendSnapshot

logger = logging.getLogger(__name__)

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"

# Mapping of technology names to their canonical GitHub repositories.
# We look up the most popular/official repo for each technology.
TECH_TO_REPO: dict[str, str] = {
    "LangChain": "langchain-ai/langchain",
    "LangGraph": "langchain-ai/langgraph",
    "LlamaIndex": "run-llama/llama_index",
    "Hugging Face Transformers": "huggingface/transformers",
    "PyTorch": "pytorch/pytorch",
    "TensorFlow": "tensorflow/tensorflow",
    "FastAPI": "tiangolo/fastapi",
    "Python": "python/cpython",
    "React": "facebook/react",
    "Next.js": "vercel/next.js",
    "Docker": "docker/docker-ce",
    "Kubernetes": "kubernetes/kubernetes",
    "Terraform": "hashicorp/terraform",
    "Apache Kafka": "apache/kafka",
    "Apache Spark": "apache/spark",
    "MLflow": "mlflow/mlflow",
    "Weaviate": "weaviate/weaviate",
    "ChromaDB": "chroma-core/chroma",
    "Qdrant": "qdrant/qdrant",
    "Pinecone": "pinecone-io/pinecone-python-client",
    "Ray": "ray-project/ray",
    "CrewAI": "crewAIInc/crewAI",
    "AutoGen": "microsoft/autogen",
    "Ollama": "ollama/ollama",
    "Groq": "groq/groq-python",
    "dbt": "dbt-labs/dbt-core",
    "Apache Airflow": "apache/airflow",
    "PostgreSQL": "postgres/postgres",
    "Redis": "redis/redis",
}


def _build_headers() -> dict[str, str]:
    """
    Build headers for GitHub API requests.

    Returns:
        Headers dict with authentication if token is available.
    """
    settings = get_settings()
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.has_github:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"
    return headers


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _fetch_repo_stats(
    client: httpx.AsyncClient,
    repo_path: str,
) -> Optional[GitHubStats]:
    """
    Fetch statistics for a specific GitHub repository.

    Args:
        client: Shared httpx async client.
        repo_path: Repository path, e.g. 'langchain-ai/langchain'.

    Returns:
        GitHubStats object, or None if the fetch fails.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo_path}"

    try:
        response = await client.get(url, headers=_build_headers())

        if response.status_code == 404:
            logger.warning("GitHub repo not found", extra={"repo": repo_path})
            return None

        response.raise_for_status()
        data = response.json()

        # Parse release date if available
        last_release = None
        release_url = f"{GITHUB_API_BASE}/repos/{repo_path}/releases/latest"
        release_resp = await client.get(release_url, headers=_build_headers())
        if release_resp.status_code == 200:
            release_data = release_resp.json()
            published = release_data.get("published_at")
            if published:
                last_release = datetime.fromisoformat(published.replace("Z", "+00:00"))

        return GitHubStats(
            repository=repo_path,
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            open_issues=data.get("open_issues_count", 0),
            contributors=None,  # Requires a separate API call — we skip to save quota
            last_release=last_release,
            language=data.get("language"),
            license=data.get("license", {}).get("spdx_id") if data.get("license") else None,
            source_url=f"https://github.com/{repo_path}",
        )

    except httpx.HTTPError as e:
        logger.error("GitHub API error", extra={"repo": repo_path, "error": str(e)})
        return None


async def _search_github_topics(
    client: httpx.AsyncClient,
    topic: str,
) -> Optional[int]:
    """
    Count repositories tagged with a specific topic.

    This is a secondary signal for technology adoption.

    Args:
        client: Shared httpx async client.
        topic: GitHub topic string (e.g., 'langchain', 'mcp').

    Returns:
        Number of repositories with this topic, or None if the API call
        failed. None is distinct from 0 — a failed request must never be
        reported as "zero repos found", since that reads as real (if
        unpopular) data rather than a missing measurement.
    """
    url = f"{GITHUB_API_BASE}/search/repositories"
    params = {"q": f"topic:{topic.lower().replace(' ', '-')}", "per_page": 1}

    try:
        response = await client.get(url, headers=_build_headers(), params=params)
        if response.status_code == 200:
            return response.json().get("total_count", 0)
        logger.error(
            "GitHub topic search failed",
            extra={"topic": topic, "status_code": response.status_code},
        )
        return None
    except httpx.HTTPError as e:
        logger.error("GitHub topic search error", extra={"topic": topic, "error": str(e)})
        return None


@tool
async def analyze_github_trends(technologies: list[str]) -> dict:
    """
    Fetch GitHub statistics for a list of technologies.

    For each technology, retrieves:
    - Star count (community interest)
    - Fork count (active usage)
    - Last release date (project health)
    - Primary language

    Args:
        technologies: List of technology names to analyze.
                      These must match keys in TECH_TO_REPO or
                      be searchable GitHub topics.

    Returns:
        Dictionary containing GitHubStats for each technology.
    """
    logger.info(
        "GitHub trend analysis started",
        extra={"technologies": technologies},
    )

    results: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=get_settings().API_TIMEOUT) as client:
        for tech in technologies:
            # Look up the canonical repo for this technology
            repo_path = TECH_TO_REPO.get(tech)

            if repo_path:
                stats = await _fetch_repo_stats(client, repo_path)
                if stats:
                    results[tech] = stats.model_dump()
                    results[tech]["source"] = "github_api"
                    logger.debug(
                        "Fetched GitHub stats",
                        extra={"tech": tech, "stars": stats.stars},
                    )
                else:
                    results[tech] = {"error": f"Could not fetch data for {repo_path}"}
            else:
                # Fall back to topic search
                repo_count = await _search_github_topics(client, tech)
                if repo_count is None:
                    results[tech] = {
                        "technology": tech,
                        "error": f"GitHub topic search failed for '{tech}'",
                        "source": "github_topic_search",
                    }
                else:
                    results[tech] = {
                        "technology": tech,
                        "topic_repo_count": repo_count,
                        "source": "github_topic_search",
                        "note": f"No canonical repo mapped — searched as topic '{tech}'",
                    }
                logger.debug(
                    "Used topic search for unknown tech",
                    extra={"tech": tech, "repo_count": repo_count},
                )

    logger.info(
        "GitHub trend analysis complete",
        extra={"analyzed": len(results)},
    )
    return results
