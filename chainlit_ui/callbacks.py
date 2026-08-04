"""
chainlit_ui/callbacks.py
========================
Streams a LangGraph run into the Chainlit UI.

DESIGN
------
Previously every graph node produced its own top-level `cl.Message` —
classifier decision, planner plan, one message per MCP call, a status
message per agent, the final answer, and the full report — often 10+
chat bubbles for a single question. That's the "UI is not great" problem:
it buries the one thing the user actually asked for under a wall of
internal chatter.

Now the whole pipeline execution renders as ONE collapsible "Agent
reasoning" step (Chainlit's native chain-of-thought UI — exactly what
this project advertises as "visualize agent reasoning in real-time"),
with one nested sub-step per agent. Collapsed by default, so a user who
just wants the answer sees only that; a user who wants to see the
reasoning can expand it.

Only the final answer is sent as a top-level chat message. The full
report is attached to that message as a side-panel element instead of a
raw HTML `<details>` block (which silently didn't render — the project's
Chainlit config has `unsafe_allow_html = false`).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# pyrefly: ignore [missing-import]
import chainlit as cl
from langchain_core.messages import AIMessage

from chainlit_ui.formatters import format_intent, format_mcp_call, format_plan

logger = logging.getLogger(__name__)

# Friendly title + Chainlit step "type" (controls the icon/styling) for
# each graph node, shown as one row in the reasoning trace.
_NODE_DISPLAY: dict[str, tuple[str, str]] = {
    "query_classifier": ("Classifying your question", "run"),
    "planner": ("Planning the analysis", "run"),
    "market_collection": ("Collecting live job postings", "tool"),
    "skill_extraction": ("Extracting skills from postings", "tool"),
    "trend_analysis": ("Analyzing technology trends", "tool"),
    "resume_analysis": ("Analyzing your resume", "tool"),
    "comparison_agent": ("Comparing countries", "tool"),
    "recommendation": ("Building recommendations", "run"),
    "chat_responder": ("Composing a reply", "llm"),
    "final_answer": ("Composing final answer", "run"),
    "report_generation": ("Saving full report", "tool"),
}

# Nodes whose AIMessage content is just an internal status line, not
# something to duplicate into the step body (the step title already
# says what happened).
_SKIP_MESSAGE_BODY = {"chat_responder", "final_answer"}


def _step_output_for_node(node_name: str, state_update: dict[str, Any]) -> str:
    """Build the collapsible step body for one completed graph node."""
    parts: list[str] = []

    if node_name == "query_classifier" and "query_intent" in state_update:
        parts.append(format_intent(state_update["query_intent"], state_update.get("intent_confidence")))

    if node_name == "planner" and "execution_plan" in state_update:
        parts.append(format_plan(state_update["execution_plan"]))

    mcp_calls = state_update.get("mcp_calls") or []
    if mcp_calls:
        latest = mcp_calls[-1] if isinstance(mcp_calls, list) else mcp_calls
        if isinstance(latest, dict):
            parts.append(format_mcp_call(latest))

    if node_name not in _SKIP_MESSAGE_BODY:
        messages = state_update.get("messages") or []
        if messages and isinstance(messages[-1], AIMessage):
            parts.append(messages[-1].content)

    return "\n\n".join(p for p in parts if p) or "_(no additional detail)_"


async def handle_graph_stream(
    stream_async_generator,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Process the stream from LangGraph, rendering a single collapsible
    "Agent reasoning" trace (one nested step per agent) instead of
    flooding the chat with a message per step.

    Args:
        stream_async_generator: The async generator from graph.astream()

    Returns:
        (final_answer, final_report) — the concise user-facing answer and
        the full report dict (if one was generated), for the caller to
        render as the primary chat message. Either may be None if the run
        errored before producing them.
    """
    final_answer: Optional[str] = None
    final_report: Optional[dict] = None

    async with cl.Step(name="Agent reasoning", type="run", default_open=False):
        async for event in stream_async_generator:
            for node_name, state_update in event.items():
                display_name, step_type = _NODE_DISPLAY.get(
                    node_name, (node_name.replace("_", " ").title(), "run")
                )
                async with cl.Step(name=display_name, type=step_type) as step:
                    step.output = _step_output_for_node(node_name, state_update)

                if node_name == "chat_responder":
                    messages = state_update.get("messages") or []
                    if messages and isinstance(messages[-1], AIMessage):
                        final_answer = messages[-1].content

                if node_name == "final_answer":
                    final_answer = state_update.get("direct_answer") or final_answer

                if node_name == "report_generation" and state_update.get("final_report"):
                    final_report = state_update["final_report"]

    return final_answer, final_report
