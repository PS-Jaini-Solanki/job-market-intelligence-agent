"""
chainlit_ui/app.py
===================
Chainlit application entry point.

Run this with: chainlit run chainlit_ui/app.py -w
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

# pyrefly: ignore [missing-import]
import chainlit as cl

# We need to set environment variables before importing anything else if needed,
# but pydantic-settings handles it for us.
import dotenv
dotenv.load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from agents.graph import get_graph
from agents.state import create_initial_state
from chainlit_ui.callbacks import handle_graph_stream
from database.connection import get_session_factory

logger = logging.getLogger(__name__)

# Configure logging for chainlit
logging.basicConfig(level=logging.INFO)


@cl.on_chat_start
async def on_chat_start():
    """Triggered when a new chat session starts."""
    await cl.Message(
        content=(
            "👋 **Welcome to the Job Market Intelligence Agent!**\n\n"
            "I can analyze the tech job market, compare salaries, check technology trends, "
            "and review your resume against market demand.\n\n"
            "**Try asking:**\n"
            "- What skills are required for AI Engineers in India?\n"
            "- Should I learn LangGraph or CrewAI?\n"
            "- Compare AI job markets in India vs USA\n\n"
            "You can also **upload your resume** (PDF, DOCX, TXT) and ask me to analyze it."
        )
    ).send()
    # Store session state
    cl.user_session.set("resume_path", None)


@cl.on_message
async def on_message(message: cl.Message):
    """Triggered when the user sends a message."""
    query = message.content
    resume_path = cl.user_session.get("resume_path")

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
                await cl.Message(content=f"✅ Received resume: `{element.name}`").send()

    # Create progress message
    msg = cl.Message(content="⚙️ Starting analysis pipeline...")
    await msg.send()

    # Create initial state
    initial_state = create_initial_state(
        user_query=query,
        resume_path=resume_path,
        query_id=str(uuid.uuid4())
    )

    # Get compiled graph
    graph = get_graph()

    try:
        # Stream the graph execution
        stream = graph.astream(initial_state)
        await handle_graph_stream(stream, msg)
        
        # Clear resume_path from session after processing so it doesn't affect next query automatically unless desired.
        # However, usually users might want to keep it context. Let's keep it for now.
        # cl.user_session.set("resume_path", None) 

    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        await cl.Message(content=f"❌ **An error occurred during execution:**\n```\n{str(e)}\n```").send()
