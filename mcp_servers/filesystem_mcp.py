"""
mcp_servers/filesystem_mcp.py
==============================
MCP server that provides access to the local filesystem.

PURPOSE:
- Read uploaded resume files
- Save generated reports
- List uploaded files

SECURITY NOTE:
Access is restricted to approved directories only (RESUME_UPLOAD_DIR
and a reports directory). The server will refuse to read files outside
these directories to prevent path traversal attacks.

TOOLS PROVIDED:
1. read_file — Read a file's contents
2. list_directory — List files in a directory
3. save_report — Save a generated report to disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import get_settings
from mcp_servers.base import MCPServer, MCPToolCall, MCPToolResult

logger = logging.getLogger(__name__)

# Allowed directory roots for file access
# We only allow access to these directories for security
ALLOWED_DIRECTORIES = [
    Path(get_settings().RESUME_UPLOAD_DIR).resolve(),
    Path("./reports").resolve(),
    Path("./uploads").resolve(),
]


def _is_path_allowed(path: Path) -> bool:
    """
    Check if a path is within an allowed directory.

    This prevents path traversal attacks like ../../etc/passwd.

    Args:
        path: The absolute path to check.

    Returns:
        True if the path is within an allowed directory.
    """
    resolved = path.resolve()
    return any(
        resolved.is_relative_to(allowed)
        for allowed in ALLOWED_DIRECTORIES
    )


class FilesystemMCPServer(MCPServer):
    """
    MCP server providing controlled filesystem access.

    Only allows reading from and writing to approved directories.
    """

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return (
            "Filesystem MCP Server — provides tools for reading uploaded "
            "resume files and saving generated reports."
        )

    def get_tools(self) -> list[dict[str, Any]]:
        """Return all filesystem tools."""
        return [
            {
                "name": "read_file",
                "description": (
                    "Read the contents of a file (text files only). "
                    "Only allowed within the uploads and reports directories."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file",
                        }
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "list_directory",
                "description": "List all files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to list",
                        }
                    },
                    "required": ["directory_path"],
                },
            },
            {
                "name": "save_report",
                "description": "Save a generated report to the reports directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file (e.g., 'report_2025.md')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write to the file",
                        },
                    },
                    "required": ["filename", "content"],
                },
            },
        ]

    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        """Route tool calls to handlers."""
        handlers = {
            "read_file": self._read_file,
            "list_directory": self._list_directory,
            "save_report": self._save_report,
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
            return MCPToolResult(tool_name=call.tool_name, result=result_data, success=True)
        except PermissionError as e:
            return MCPToolResult(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error=str(e),
            )

    async def _read_file(self, file_path: str) -> dict:
        """Read a file if it's within an allowed directory."""
        path = Path(file_path)

        if not _is_path_allowed(path):
            raise PermissionError(
                f"Access denied: '{file_path}' is outside allowed directories"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # For text files, return content directly
        # For binary files (PDF, DOCX), indicate that a parser is needed
        if path.suffix.lower() in (".pdf", ".docx", ".doc"):
            return {
                "file_path": str(path),
                "file_type": path.suffix.lower(),
                "note": "Binary file — use parse_resume tool to extract text",
                "size_bytes": path.stat().st_size,
            }

        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "file_path": str(path),
            "content": content,
            "size_bytes": len(content),
        }

    async def _list_directory(self, directory_path: str) -> dict:
        """List files in a directory if allowed."""
        path = Path(directory_path)

        if not _is_path_allowed(path):
            raise PermissionError(
                f"Access denied: '{directory_path}' is outside allowed directories"
            )

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        files = []
        for item in path.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size_bytes": item.stat().st_size if item.is_file() else None,
                "suffix": item.suffix.lower() if item.is_file() else None,
            })

        return {"directory": str(path), "files": files, "count": len(files)}

    async def _save_report(self, filename: str, content: str) -> dict:
        """Save a report to the reports directory."""
        reports_dir = Path("./reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename — only allow alphanumeric, dash, underscore, dot
        import re
        safe_filename = re.sub(r"[^\w\-.]", "_", filename)
        output_path = reports_dir / safe_filename

        output_path.write_text(content, encoding="utf-8")

        logger.info("Report saved", extra={"path": str(output_path)})
        return {
            "saved_to": str(output_path),
            "size_bytes": len(content),
        }
