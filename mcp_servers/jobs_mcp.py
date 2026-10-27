"""
mcp_servers/jobs_mcp.py
=========================
MCP server that wraps job-market data collection tools.

TOOLS PROVIDED:
1. search_jobs — Collect real job postings from Adzuna + Arbeitnow
2. compare_countries — Compare job markets across countries

This server is used by:
- MarketCollectionAgent (search_jobs)
- Planner-directed country comparison tasks (compare_countries)

It standardizes the job search capability behind the MCP interface so
the Planner can declare "jobs" as a required MCP server for a task and
any agent can invoke it through the registry.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mcp_servers.base import MCPServer, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)


class JobsMCPServer(MCPServer):
    """
    MCP server providing real-time job posting collection.

    Uses two free job board APIs:
    - Adzuna (requires ADZUNA_APP_ID + ADZUNA_APP_KEY)
    - Arbeitnow (no key required)
    """

    @property
    def name(self) -> str:
        return "jobs"

    @property
    def description(self) -> str:
        return (
            "Jobs MCP Server — collects real-time job postings from "
            "Adzuna and Arbeitnow and compares job markets across countries."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Return all job search tools."""
        return [
            {
                "name": "search_jobs",
                "description": (
                    "Search for real job postings from Adzuna and Arbeitnow. "
                    "Returns postings with title, company, location, salary and description."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "Job title to search for (e.g., 'AI Engineer')",
                        },
                        "country": {
                            "type": "string",
                            "description": "ISO country code (e.g., 'IN', 'US', 'GB'). Omit for no country filter",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of postings to collect (default: 100)",
                        },
                    },
                    "required": ["role"],
                },
            },
            {
                "name": "compare_countries",
                "description": (
                    "Compare job market conditions (skills, salaries, volume) "
                    "across multiple countries for a given role."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "Job role to compare (e.g., 'AI Engineer')",
                        },
                        "country_codes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "ISO country codes to compare (e.g., ['IN', 'US'])",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Postings per country (default: 50)",
                        },
                        "job_search_result": {
                            "type": "object",
                            "description": (
                                "Optional postings already collected for this role. "
                                "When given, no new API calls are made."
                            ),
                        },
                    },
                    "required": ["role", "country_codes"],
                },
            },
        ]

    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """Route tool calls to handlers."""
        handlers = {
            "search_jobs": self._search_jobs,
            "compare_countries": self._compare_countries,
        }

        handler = handlers.get(call.tool_name)
        if not handler:
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=f"Unknown tool: {call.tool_name}",
            )

        try:
            result_data = await handler(**call.arguments)
            return MCPToolResult(
                tool_name=call.tool_name,
                result=result_data,
                success=True,
            )
        except Exception as e:
            logger.error("Jobs MCP tool failed", extra={"tool": call.tool_name, "error": str(e)})
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=str(e),
            )

    async def _search_jobs(
        self,
        role: str,
        country: Optional[str] = None,
        max_results: int = 100,
    ) -> dict:
        """
        Collect job postings from Adzuna and Arbeitnow.

        Args:
            role: Job title to search for.
            country: ISO country code, or None for no country filter.
            max_results: Total postings budget.

        Returns:
            Serialized JobSearchResult dictionary.
        """
        # Imported here to avoid a heavy import chain at module load
        from tools.job_search_tool import collect_postings

        return (await collect_postings(role, country, max_results)).model_dump()

    async def _compare_countries(
        self,
        role: str,
        country_codes: list[str],
        max_results: int = 50,
        job_search_result: Optional[dict] = None,
    ) -> dict:
        """
        Compare job markets across multiple countries.

        Reuses postings the caller already collected when given; otherwise
        collects them per country. Analysis is delegated to the
        country_comparison_tool.

        Args:
            role: Job role to compare.
            country_codes: List of ISO country codes.
            max_results: Postings budget per country (only when collecting).
            job_search_result: Optional serialized JobSearchResult to reuse.

        Returns:
            Comparison dictionary.
        """
        from models.job import JobSearchResult
        from tools.country_comparison_tool import compare_countries
        from tools.job_search_tool import collect_postings

        if job_search_result is None:
            postings = []
            sources: list[str] = []
            for code in country_codes:
                partial = await collect_postings(role, code, max_results)
                postings.extend(partial.postings)
                sources.extend(s for s in partial.sources_queried if s not in sources)
            job_search_result = JobSearchResult(
                query=role,
                postings=postings,
                total_found=len(postings),
                sources_queried=sources,
                search_params={"role": role},
            ).model_dump()

        return await compare_countries.ainvoke(
            {"job_search_result": job_search_result, "country_codes": country_codes}
        )
