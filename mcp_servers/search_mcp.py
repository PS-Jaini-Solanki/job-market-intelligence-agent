"""
mcp_servers/search_mcp.py
==========================
MCP server wrapping the Tavily Search API.

PURPOSE:
- Search the web for job market information
- Find tech blog articles about emerging tools
- Search for salary reports and developer surveys

TAVILY vs GOOGLE:
Tavily is designed specifically for AI agents — it returns clean,
structured results that are easy for LLMs to process. Regular Google
search returns HTML that needs heavy parsing.

TOOLS PROVIDED:
1. web_search — General web search
2. search_tech_news — Search for technology news and blog posts
3. search_market_reports — Search for salary and job market reports
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import get_settings
from mcp_servers.base import MCPServer, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)


class SearchMCPServer(MCPServer):
    """
    MCP server providing web search capabilities via Tavily.

    Requires TAVILY_API_KEY in .env (free tier: 1000 calls/month).
    """

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search MCP Server — provides web search tools for finding "
            "job market data, tech articles, and salary reports."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Return all search tools."""
        return [
            {
                "name": "web_search",
                "description": "Search the web for any information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 5)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_tech_news",
                "description": "Search for technology news and developer blog posts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology to search for (e.g., 'MCP', 'LangGraph')",
                        },
                        "recency_days": {
                            "type": "integer",
                            "description": "Only return results from the last N days (default: 30)",
                        },
                    },
                    "required": ["technology"],
                },
            },
            {
                "name": "search_market_reports",
                "description": "Search for salary surveys and developer market reports.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "Job role to search for (e.g., 'AI Engineer')",
                        },
                        "country": {
                            "type": "string",
                            "description": "Country to focus on (optional)",
                        },
                    },
                    "required": ["role"],
                },
            },
        ]

    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """Route tool calls to handlers."""
        settings = get_settings()
        if not settings.has_tavily:
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error="TAVILY_API_KEY not configured. Add it to your .env file.",
            )

        handlers = {
            "web_search": self._web_search,
            "search_tech_news": self._search_tech_news,
            "search_market_reports": self._search_market_reports,
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
        return MCPToolResult(tool_name=call.tool_name, result=result_data, success=True)

    async def _tavily_search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Execute a Tavily search query.

        Args:
            query: Search query string.
            max_results: Max results to return.

        Returns:
            List of result dictionaries.
        """
        # pyrefly: ignore [missing-import]
        from tavily import TavilyClient

        settings = get_settings()
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # 'basic' uses fewer API credits than 'advanced'
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content", "")[:500],  # Truncate to 500 chars
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            })

        return results

    async def _web_search(self, query: str, max_results: int = 5) -> dict:
        """General web search."""
        logger.info("Web search", extra={"query": query})
        results = await self._tavily_search(query, max_results)
        return {"query": query, "results": results, "source": "tavily"}

    async def _search_tech_news(
        self, technology: str, recency_days: int = 30
    ) -> dict:
        """Search for technology news."""
        query = f"{technology} tutorial blog article 2024 2025"
        logger.info("Tech news search", extra={"technology": technology})
        results = await self._tavily_search(query, max_results=5)
        return {
            "technology": technology,
            "results": results,
            "source": "tavily",
            "note": "Results filtered for relevance by Tavily search engine.",
        }

    async def _search_market_reports(
        self, role: str, country: Optional[str] = None
    ) -> dict:
        """Search for salary and market report data."""
        location_str = f" {country}" if country else ""
        query = (
            f"{role}{location_str} salary survey 2024 2025 "
            "developer market report average compensation"
        )
        logger.info("Market report search", extra={"role": role, "country": country})
        results = await self._tavily_search(query, max_results=5)
        return {
            "role": role,
            "country": country,
            "results": results,
            "source": "tavily",
            "important_note": (
                "These are web search results — not direct salary data. "
                "For verified salary data, use the parse_salaries tool "
                "on actual job postings."
            ),
        }
