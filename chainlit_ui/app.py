"""
chainlit_ui/app.py
===================
Chainlit application entry point.

Run this with: chainlit run chainlit_ui/app.py -w
"""

from __future__ import annotations

# ============================================================
# CRITICAL: Add the project root to sys.path FIRST.
# Chainlit runs this file from chainlit_ui/ so Python won't
# find top-level packages (agents/, mcp_servers/, tools/, etc.)
# unless we explicitly add the parent directory here.
# This MUST happen before any project imports.
#
# WHY WE RE-ASSERT AT RUNTIME:
# Chainlit's load_module() inserts the app's directory into
# sys.path, executes this file, then POPS the entry it added.
# That pop can silently remove the project root we insert here.
# Module-level imports still work (they are cached in sys.modules)
# but the lazy imports agents do at runtime — e.g.
# `from mcp_servers.registry import get_registry` — fail with
# "No module named 'mcp_servers'". So we re-assert the root right
# before every graph execution too.
# ============================================================
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    """Ensure the project root is importable from any launcher/worker."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root_on_path()

import logging
import os
import shutil
import uuid

# Load .env before importing config/settings
import dotenv
dotenv.load_dotenv(str(Path(__file__).resolve().parent.parent) + "/.env")

# pyrefly: ignore [missing-import]
import chainlit as cl

from agents.graph import get_graph
from agents.state import create_initial_state
from chainlit_ui.callbacks import handle_graph_stream
from chainlit_ui.formatters import format_market_report
from database.connection import get_session_factory

logger = logging.getLogger(__name__)

# Configure logging for chainlit
logging.basicConfig(level=logging.INFO)


@cl.on_chat_start
async def on_chat_start():
    """Triggered when a new chat session starts."""
    _ensure_project_root_on_path()
    await cl.Message(
        content=(
            "**Job Market Intelligence Agent**\n\n"
            "I analyze real job postings, GitHub trends, and (if you upload one) "
            "your resume — every number I give you comes from live data, never a guess.\n\n"
            "Try asking:\n"
            "- What skills are required for AI Engineers in India?\n"
            "- Should I learn LangGraph or CrewAI?\n"
            "- Compare AI job markets in India vs USA\n"
            "- What's the average salary of a Data Scientist in the UK?\n\n"
            "You can also upload a resume (PDF, DOCX, or TXT) for a gap analysis "
            "against the market. Follow-ups like *\"what about salaries in India?\"* "
            "carry context from earlier in the chat automatically.\n\n"
            "Every answer has an **Agent reasoning** panel above it — expand it "
            "to see exactly which agents ran and which live data sources were called."
        )
    ).send()
    # Store session state
    cl.user_session.set("resume_path", None)
    cl.user_session.set("conversation_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """Triggered when the user sends a message."""
    _ensure_project_root_on_path()

    query = message.content
    resume_path = cl.user_session.get("resume_path")
    conversation_history = cl.user_session.get("conversation_history") or []

    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if element.type == "file":
                # Save the uploaded file
                upload_dir = Path("./uploads/resumes")
                upload_dir.mkdir(parents=True, exist_ok=True)
                file_path = upload_dir / f"{uuid.uuid4()}_{element.name}"

                # Copy file from Chainlit's temporary location to our uploads dir
                with open(element.path, "rb") as f_in:
                    with open(file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                resume_path = str(file_path)
                cl.user_session.set("resume_path", resume_path)
                await cl.Message(
                    content=(
                        f"Got it — **{element.name}** is saved for this conversation. "
                        "It'll be used for resume/gap-analysis questions; it won't affect "
                        "unrelated questions like salary or trend lookups."
                    )
                ).send()

    # Create initial state. Note: resume_path may be set from an earlier
    # turn in this session — the graph only actually runs resume analysis
    # (agents/graph.py: should_analyze_resume) when this turn's classified
    # intent is resume-relevant, so carrying it forward doesn't hijack
    # unrelated questions.
    initial_state = create_initial_state(
        user_query=query,
        resume_path=resume_path,
        query_id=str(uuid.uuid4()),
        conversation_history=conversation_history,
    )

    # Get compiled graph
    graph = get_graph()

    try:
        # Stream the graph execution. The reasoning trace renders as a
        # single collapsible step (see callbacks.py); only the final
        # answer becomes a top-level chat message.
        stream = graph.astream(initial_state)
        assistant_answer, final_report = await handle_graph_stream(stream)

        if not assistant_answer:
            assistant_answer = (
                "I wasn't able to produce an answer for that — could you rephrase "
                "or try again in a moment?"
            )

        elements = []
        if final_report:
            report_md = format_market_report(final_report)
            if not report_md.startswith("Error formatting report"):
                elements.append(
                    cl.Text(name="Full report", content=report_md, display="side")
                )

        await cl.Message(content=assistant_answer, author="Assistant", elements=elements).send()

        # Remember this turn so follow-up questions keep context
        conversation_history = [*conversation_history, {"role": "user", "content": query}]
        conversation_history.append({"role": "assistant", "content": assistant_answer})
        cl.user_session.set("conversation_history", conversation_history[-12:])

    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        await cl.Message(
            content=(
                "Something went wrong while working on that.\n\n"
                f"```\n{str(e)}\n```"
            )
        ).send()
