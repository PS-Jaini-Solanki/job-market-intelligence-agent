"""
mcp_servers/base.py
====================
Abstract base class for all MCP (Model Context Protocol) servers.

WHAT IS MCP?
MCP is a protocol that lets AI models communicate with external tools
in a standardized way. Think of it like a "USB standard for AI tools":
- Any MCP server exposes tools in the same format.
- Any MCP client (LLM) can call any MCP server's tools.
- No custom integration code needed for each tool.

In this project, we wrap our own tools as MCP servers so they can be
called by the LangGraph agents in a standardized way.

ARCHITECTURE:
  LangGraph Agent
      │
      └── MCPRegistry
              │
              ├── GitHubMCPServer
              ├── FilesystemMCPServer
              ├── SearchMCPServer
              └── RedditMCPServer

Each MCPServer wraps one or more tools and provides them through
the standardized call_tool() interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPToolCall:
    """
    Represents a request to call a tool on an MCP server.

    This is the standardized input format for all MCP tool calls.
    """

    tool_name: str          # Name of the tool to call
    arguments: dict[str, Any]  # Input arguments for the tool
    request_id: Optional[str] = None  # Optional ID for tracking


@dataclass
class MCPToolResult:
    """
    Represents the result of an MCP tool call.

    This is the standardized output format for all MCP tool calls.
    """

    tool_name: str
    result: Any             # The tool's output (can be any JSON-serializable type)
    success: bool           # True if the call succeeded
    error: Optional[str] = None   # Error message if success=False
    execution_time_ms: Optional[int] = None  # How long the call took


class MCPServer(ABC):
    """
    Abstract base class for all MCP servers in this project.

    To create a new MCP server:
    1. Subclass MCPServer.
    2. Implement get_tools() to return available tool definitions.
    3. Implement call_tool() to execute tool calls.
    4. Register in mcp_servers/registry.py.

    Example:
        class MyMCPServer(MCPServer):
            @property
            def name(self) -> str:
                return "my_server"

            def get_tools(self) -> list[dict]:
                return [{"name": "my_tool", "description": "..."}]

            async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
                # Execute the tool
                result = await do_something(call.arguments)
                return MCPToolResult(
                    tool_name=call.tool_name,
                    result=result,
                    success=True,
                )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The unique name of this MCP server.

        Used by the registry to look up servers.
        Example: "github", "filesystem", "search"
        """
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this server provides."""
        return f"MCP Server: {self.name}"

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return the list of tools this server provides.

        Each tool is described by a dictionary with:
        - name: Tool identifier
        - description: What the tool does
        - parameters: Input parameter schema (JSON Schema format)

        Returns:
            List of tool definition dictionaries.
        """
        ...

    @abstractmethod
    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """
        Execute a tool call on this server.

        Args:
            call: The MCPToolCall specifying which tool to call and with what args.

        Returns:
            MCPToolResult with the tool's output.
        """
        ...

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if this server provides a specific tool.

        Args:
            tool_name: Name of the tool to look for.

        Returns:
            True if this server has the named tool.
        """
        return any(t["name"] == tool_name for t in self.get_tools())

    async def safe_call(self, call: MCPToolCall) -> MCPToolResult:
        """
        Call a tool with error handling and logging.

        This wraps call_tool() with:
        - Validation that the tool exists
        - Timing measurement
        - Error catching and logging

        Args:
            call: The MCPToolCall to execute.

        Returns:
            MCPToolResult (success=False if an error occurred).
        """
        import time

        if not self.has_tool(call.tool_name):
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=f"Tool '{call.tool_name}' not found on server '{self.name}'",
            )

        start = time.monotonic()
        logger.info(
            "MCP tool call started",
            extra={
                "server": self.name,
                "tool": call.tool_name,
                "request_id": call.request_id,
            },
        )

        try:
            result = await self.call_tool(call)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            result.execution_time_ms = elapsed_ms

            logger.info(
                "MCP tool call completed",
                extra={
                    "server": self.name,
                    "tool": call.tool_name,
                    "success": result.success,
                    "elapsed_ms": elapsed_ms,
                },
            )
            return result

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "MCP tool call failed",
                extra={
                    "server": self.name,
                    "tool": call.tool_name,
                    "error": str(e),
                    "elapsed_ms": elapsed_ms,
                },
            )
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
