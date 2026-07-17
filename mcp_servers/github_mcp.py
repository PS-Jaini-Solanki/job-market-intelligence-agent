"""
mcp_servers/github_mcp.py
==========================
MCP server that wraps GitHub API tools.

TOOLS PROVIDED:
1. get_repo_stats — Fetch stats for a specific repository
2. search_repositories — Search GitHub repositories by keyword
3. get_trending — Get trending repositories for a topic

This server is used by:
- TrendAnalysisAgent (to assess technology popularity)
- PlannerAgent (to decide if GitHub data is worth collecting)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import get_settings
from mcp_servers.base import MCPServer, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class GitHubMCPServer(MCPServer):
    """
    MCP server providing GitHub repository analysis tools.

    Requires GITHUB_TOKEN in .env for higher rate limits.
    Without a token, limited to 60 requests/hour.
    """

    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return (
            "GitHub MCP Server — provides tools for analyzing repository "
            "popularity, contributor activity, and technology trends."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Return all GitHub tools available on this server."""
        return [
            {
                "name": "get_repo_stats",
                "description": (
                    "Fetch statistics for a GitHub repository. "
                    "Returns star count, fork count, language, and recent activity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path, e.g. 'langchain-ai/langchain'",
                        }
                    },
                    "required": ["repo_path"],
                },
            },
            {
                "name": "search_repositories",
                "description": (
                    "Search GitHub for repositories matching a keyword. "
                    "Returns a list of repositories sorted by stars."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Search keyword (e.g., 'mcp server', 'langgraph')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 10)",
                        },
                    },
                    "required": ["keyword"],
                },
            },
            {
                "name": "get_trending",
                "description": (
                    "Get trending repositories for a programming language or topic. "
                    "Uses GitHub's search API sorted by recent stars."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to search (e.g., 'mcp', 'llm', 'agentic-ai')",
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language filter (optional)",
                        },
                    },
                    "required": ["topic"],
                },
            },
        ]

    def _get_headers(self) -> dict[str, str]:
        """Build GitHub API request headers."""
        settings = get_settings()
        headers = {"Accept": "application/vnd.github.v3+json"}
        if settings.has_github:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"
        return headers

    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """Route tool calls to the appropriate handler."""
        handlers = {
            "get_repo_stats": self._get_repo_stats,
            "search_repositories": self._search_repositories,
            "get_trending": self._get_trending,
        }

        handler = handlers.get(call.tool_name)
        if not handler:
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=f"Unknown tool: {call.tool_name}",
            )

        result_data = await handler(**call.arguments)
        return MCPToolResult(
            tool_name=call.tool_name,
            result=result_data,
            success=True,
        )

    async def _get_repo_stats(self, repo_path: str) -> dict:
        """Fetch statistics for a specific repository."""
        url = f"{GITHUB_API_BASE}/repos/{repo_path}"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self._get_headers())

            if response.status_code == 404:
                return {"error": f"Repository '{repo_path}' not found"}
            if response.status_code == 403:
                return {"error": "GitHub rate limit exceeded. Add GITHUB_TOKEN to .env"}

            response.raise_for_status()
            data = response.json()

            return {
                "repository": repo_path,
                "full_name": data.get("full_name"),
                "description": data.get("description"),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
                "language": data.get("language"),
                "license": data.get("license", {}).get("spdx_id") if data.get("license") else None,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "topics": data.get("topics", []),
                "source_url": f"https://github.com/{repo_path}",
            }

    async def _search_repositories(self, keyword: str, limit: int = 10) -> dict:
        """Search GitHub repositories by keyword."""
        url = f"{GITHUB_API_BASE}/search/repositories"
        params = {
            "q": keyword,
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 30),
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            data = response.json()

            repos = []
            for item in data.get("items", []):
                repos.append({
                    "name": item.get("full_name"),
                    "description": item.get("description"),
                    "stars": item.get("stargazers_count", 0),
                    "language": item.get("language"),
                    "url": item.get("html_url"),
                    "topics": item.get("topics", []),
                })

            return {
                "keyword": keyword,
                "total_found": data.get("total_count", 0),
                "repositories": repos,
            }

    async def _get_trending(self, topic: str, language: str = "") -> dict:
        """Get trending repositories for a topic."""
        query = f"topic:{topic.lower().replace(' ', '-')}"
        if language:
            query += f" language:{language}"

        url = f"{GITHUB_API_BASE}/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 10,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            data = response.json()

            repos = [
                {
                    "name": item.get("full_name"),
                    "stars": item.get("stargazers_count", 0),
                    "description": item.get("description"),
                    "url": item.get("html_url"),
                }
                for item in data.get("items", [])
            ]

            return {
                "topic": topic,
                "total_repos_with_topic": data.get("total_count", 0),
                "top_repositories": repos,
            }
