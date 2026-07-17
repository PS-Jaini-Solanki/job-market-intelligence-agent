"""
chainlit_ui/callbacks.py
========================
Callbacks for LangGraph execution to update the Chainlit UI.

This file handles streaming updates from the agent graph to the frontend.
"""

from __future__ import annotations

import logging
from typing import Any

# pyrefly: ignore [missing-import]
import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from chainlit_ui.formatters import format_plan, format_market_report

logger = logging.getLogger(__name__)


async def handle_graph_stream(stream_async_generator, message_ui: cl.Message) -> None:
    """
    Process the stream from LangGraph and update the UI.

    Args:
        stream_async_generator: The async generator from graph.astream()
        message_ui: The initial Chainlit message to update
    """
    content_buffer = []

    async for event in stream_async_generator:
        # LangGraph returns a dict where the key is the node name
        # and the value is the partial state update
        for node_name, state_update in event.items():
            # Add a step indicator to the UI message
            content_buffer.append(f"\n✅ **Completed:** `{node_name}`")
            await message_ui.stream_token(f"\n✅ **Completed:** `{node_name}`")

            # Look for specific updates to show in UI
            if node_name == "planner" and "execution_plan" in state_update:
                plan_md = format_plan(state_update["execution_plan"])
                # Send plan as a separate message for clarity
                await cl.Message(content=plan_md, author="Planner Agent").send()

            # Handle new messages added to state
            messages = state_update.get("messages", [])
            if messages:
                # We only want to show the latest message added by this node
                latest_msg = messages[-1]
                if isinstance(latest_msg, AIMessage):
                    await cl.Message(content=latest_msg.content, author=f"{node_name.replace('_', ' ').title()}").send()

            # Handle final report
            if node_name == "report_generation" and "final_report" in state_update:
                report_md = format_market_report(state_update["final_report"])
                await cl.Message(content=report_md, author="Final Report").send()

    # Final update to the progress message
    await message_ui.update()
