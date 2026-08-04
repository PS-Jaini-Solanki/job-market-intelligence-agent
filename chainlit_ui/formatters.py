"""
chainlit_ui/formatters.py
=========================
Formatters for the Chainlit UI.

This file contains helper functions to format agent messages, tool calls,
and reports into beautiful Markdown for the Chainlit interface.
"""

from __future__ import annotations

import json
from typing import Any

from models.agent import MarketReport


def format_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """Format an MCP tool call for display."""
    args_json = json.dumps(arguments, indent=2)
    return f"**🔧 Calling Tool:** `{tool_name}`\n```json\n{args_json}\n```"


def format_tool_result(tool_name: str, result: Any, success: bool, execution_time_ms: int | None = None) -> str:
    """Format an MCP tool result for display."""
    time_str = f" ({execution_time_ms}ms)" if execution_time_ms else ""
    if success:
        if isinstance(result, (dict, list)):
            result_json = json.dumps(result, indent=2)
            # Truncate if too long
            if len(result_json) > 1000:
                result_json = result_json[:1000] + "\n... [truncated]"
            return f"**✅ Tool Succeeded:** `{tool_name}`{time_str}\n```json\n{result_json}\n```"
        return f"**✅ Tool Succeeded:** `{tool_name}`{time_str}\n```\n{result}\n```"
    else:
        return f"**❌ Tool Failed:** `{tool_name}`{time_str}\n```\n{result}\n```"


def format_market_report(report_dict: dict) -> str:
    """Format a MarketReport dictionary into Markdown."""
    try:
        report = MarketReport.model_validate(report_dict)
        return report.to_markdown()
    except Exception as e:
        return f"Error formatting report: {str(e)}\n\n```json\n{json.dumps(report_dict, indent=2)}\n```"


def format_plan(plan_dict: dict) -> str:
    """Format a planner output dictionary into Markdown."""
    tasks = plan_dict.get("tasks", [])
    reasoning = plan_dict.get("reasoning", "No reasoning provided.")
    query_type = plan_dict.get("query_type", "Unknown")

    md = f"### 🧠 Execution Plan\n**Query Type:** `{query_type}`\n\n**Reasoning:** {reasoning}\n\n**Tasks:**\n"
    for i, task in enumerate(tasks, 1):
        md += f"{i}. **{task.get('task_type')}**\n"
        params = task.get("parameters", {})
        if params:
            md += f"   - Parameters: `{json.dumps(params)}`\n"
        mcp = task.get("required_mcp_servers", [])
        if mcp:
            md += f"   - Required MCP: `{', '.join(mcp)}`\n"

    return md


def format_mcp_call(call_dict: dict) -> str:
    """Format a recorded MCP call into a compact Markdown line."""
    try:
        from models.agent import MCPCallRecord
        record = MCPCallRecord.model_validate(call_dict)
        return record.to_markdown()
    except Exception:
        status = "✅" if call_dict.get("success", True) else "❌"
        args = ", ".join(
            f"{k}={v}" for k, v in list(call_dict.get("arguments", {}).items())[:3]
        )
        return (
            f"{status} **{call_dict.get('agent', '?')}** → "
            f"`{call_dict.get('server', '?')}` MCP · "
            f"`{call_dict.get('tool', '?')}({args})`"
        )


def format_intent(intent: str, confidence: float) -> str:
    """Format the classifier's intent decision for display."""
    conf = f" ({confidence:.0%} confidence)" if confidence else ""
    return f"🧭 **Intent detected:** `{intent}`{conf}"
