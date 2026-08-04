"""
mcp_servers/registry.py
========================
Central registry of all available MCP servers.

The registry serves two purposes:
1. Discovery — agents can ask "what servers are available?"
2. Routing — agents can call tools on any registered server.

The Planner agent uses this registry to decide which servers
are needed for a given query and which tools to call.

USAGE:
    from mcp_servers.registry import MCPRegistry

    registry = MCPRegistry()
    result = await registry.call_tool("github", "get_repo_stats", {
        "repo_path": "langchain-ai/langchain"
    })
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mcp_servers.base import MCPServer, MCPToolCall, MCPToolResult
from mcp_servers.github_mcp import GitHubMCPServer
from mcp_servers.filesystem_mcp import FilesystemMCPServer
from mcp_servers.search_mcp import SearchMCPServer
from mcp_servers.reddit_mcp import RedditMCPServer
from mcp_servers.jobs_mcp import JobsMCPServer

logger = logging.getLogger(__name__)


class MCPRegistry:
    """
    Registry of all available MCP servers.

    The registry is a singleton — use get_registry() to get the
    shared instance rather than creating new ones.
    """

    def __init__(self) -> None:
        # Initialize all available servers
        self._servers: dict[str, MCPServer] = {
            "github": GitHubMCPServer(),
            "filesystem": FilesystemMCPServer(),
            "search": SearchMCPServer(),
            "reddit": RedditMCPServer(),
            "jobs": JobsMCPServer(),
        }
        logger.info(
            "MCP Registry initialized",
            extra={"servers": list(self._servers.keys())},
        )

    def get_server(self, server_name: str) -> Optional[MCPServer]:
        """
        Get a specific MCP server by name.

        Args:
            server_name: Name of the server (e.g., 'github', 'search').

        Returns:
            The MCPServer instance, or None if not found.
        """
        return self._servers.get(server_name)

    def list_servers(self) -> list[dict[str, Any]]:
        """
        List all registered servers and their tools.

        Returns:
            List of server info dictionaries, each containing:
            - name: Server identifier
            - description: What the server does
            - tools: List of tool names available on this server
        """
        return [
            {
                "name": server.name,
                "description": server.description,
                "tools": [t["name"] for t in server.get_tools()],
            }
            for server in self._servers.values()
        ]

    def get_all_tool_names(self) -> list[str]:
        """
        Get a flat list of all tool names across all servers.

        Returns:
            List of strings in "server.tool" format.
            Example: ["github.get_repo_stats", "search.web_search"]
        """
        tool_names = []
        for server in self._servers.values():
            for tool in server.get_tools():
                tool_names.append(f"{server.name}.{tool['name']}")
        return tool_names

    def find_tool(self, tool_name: str) -> Optional[tuple[str, str]]:
        """
        Find which server provides a given tool.

        Args:
            tool_name: Either "server.tool" format or just "tool".

        Returns:
            (server_name, tool_name) tuple, or None if not found.
        """
        # Handle "server.tool" format
        if "." in tool_name:
            server_name, tool = tool_name.split(".", 1)
            server = self._servers.get(server_name)
            if server and server.has_tool(tool):
                return (server_name, tool)
            return None

        # Search all servers for the tool name
        for server in self._servers.values():
            if server.has_tool(tool_name):
                return (server.name, tool_name)

        return None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        request_id: Optional[str] = None,
    ) -> MCPToolResult:
        """
        Call a tool on a specific server.

        This is the main entry point for agent code to call MCP tools.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Input arguments for the tool.
            request_id: Optional request ID for logging.

        Returns:
            MCPToolResult with the tool's output.
        """
        server = self.get_server(server_name)
        if not server:
            return MCPToolResult(
                tool_name=tool_name,
                result=None,
                success=False,
                error=f"MCP server '{server_name}' not found in registry",
            )

        call = MCPToolCall(
            tool_name=tool_name,
            arguments=arguments,
            request_id=request_id,
        )

        return await server.safe_call(call)

    async def call_tool_auto_route(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        request_id: Optional[str] = None,
    ) -> MCPToolResult:
        """
        Call a tool without specifying the server — auto-routes.

        Useful when the planner knows WHAT tool to call but not WHICH server.

        Args:
            tool_name: Tool name (either "server.tool" or just "tool").
            arguments: Input arguments.
            request_id: Optional request ID.

        Returns:
            MCPToolResult with output.
        """
        location = self.find_tool(tool_name)
        if not location:
            return MCPToolResult(
                tool_name=tool_name,
                result=None,
                success=False,
                error=f"Tool '{tool_name}' not found on any registered MCP server",
            )

        server_name, resolved_tool_name = location
        return await self.call_tool(
            server_name=server_name,
            tool_name=resolved_tool_name,
            arguments=arguments,
            request_id=request_id,
        )


# Module-level singleton
_registry: Optional[MCPRegistry] = None


def get_registry() -> MCPRegistry:
    """
    Return the shared MCP registry instance.

    Creates it on first call (lazy initialization).
    """
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry
