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
                            "description": "ISO country code (e.g., 'IN', 'US', 'GB'). Default: 'US'",
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
        country: str = "US",
        max_results: int = 100,
    ) -> dict:
        """
        Collect job postings from Adzuna and Arbeitnow.

        Args:
            role: Job title to search for.
            country: ISO country code.
            max_results: Total postings budget (split across sources).

        Returns:
            Serialized JobSearchResult dictionary.
        """
        # Imported here to avoid a heavy import chain at module load
        from tools.job_search_tool import _search_adzuna, _search_arbeitnow
        from models.job import JobSearchResult

        per_source = max(1, max_results // 2)

        adzuna_jobs = await _search_adzuna(
            role=role,
            country_code=country,
            max_results=per_source,
        )
        arbeitnow_jobs = await _search_arbeitnow(
            role=role,
            max_results=per_source,
        )

        all_postings = adzuna_jobs + arbeitnow_jobs
        sources = []
        if adzuna_jobs:
            sources.append("adzuna")
        if arbeitnow_jobs:
            sources.append("arbeitnow")

        result = JobSearchResult(
            query=role,
            postings=all_postings,
            total_found=len(all_postings),
            sources_queried=sources,
            search_params={"role": role, "country": country},
        )

        logger.info(
            "Jobs MCP search complete",
            extra={"role": role, "country": country, "count": len(all_postings), "sources": sources},
        )
        return result.model_dump()

    async def _compare_countries(
        self,
        role: str,
        country_codes: list[str],
        max_results: int = 50,
    ) -> dict:
        """
        Compare job markets across multiple countries.

        Collects postings per country and delegates to the
        country_comparison_tool for analysis.

        Args:
            role: Job role to compare.
            country_codes: List of ISO country codes.
            max_results: Postings budget per country.

        Returns:
            Comparison dictionary.
        """
        from tools.job_search_tool import _search_adzuna, _search_arbeitnow
        from models.job import JobSearchResult
        from tools.country_comparison_tool import compare_countries

        all_postings = []
        for code in country_codes:
            jobs = await _search_adzuna(
                role=role,
                country_code=code,
                max_results=max_results,
            )
            all_postings.extend(jobs)

        # Arbeitnow has no country parameter — it returns a single pool of
        # postings with the country inferred from each posting's location.
        # Previously this method only queried Adzuna, so comparisons
        # silently under-counted the market (especially for European
        # countries, where Arbeitnow has strong coverage) relative to the
        # regular single-country search_jobs path, which does query both.
        arbeitnow_jobs = await _search_arbeitnow(role=role, max_results=max_results)
        wanted_codes = {c.upper() for c in country_codes}
        arbeitnow_matches = [j for j in arbeitnow_jobs if j.country in wanted_codes]
        all_postings.extend(arbeitnow_matches)

        sources_queried = ["adzuna"] + (["arbeitnow"] if arbeitnow_matches else [])
        job_result = JobSearchResult(
            query=role,
            postings=all_postings,
            total_found=len(all_postings),
            sources_queried=sources_queried,
            search_params={"role": role},
        )

        comparison = await compare_countries.ainvoke(
            {
                "job_search_result": job_result.model_dump(),
                "country_codes": country_codes,
            }
        )
        return comparison
