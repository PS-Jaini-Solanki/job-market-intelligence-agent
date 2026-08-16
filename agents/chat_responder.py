"""
agents/chat_responder.py
=========================
ChatResponder Agent — answers conversational, general and off-topic questions.

WHY IT EXISTS
-------------
Previously EVERY user query ran the full market-analysis pipeline and
produced a big report — even for questions like "hi" or "what is the
weather like?". That produced identical-looking reports for unrelated
questions and gave users a poor experience.

This agent handles everything the pipeline does NOT need to run for:
- Greetings, thanks, capability questions (fast templates, no LLM)
- Casual / general questions (LLM answer)
- Factual questions (optionally grounded with the search MCP)
- Off-topic questions (brief answer + gentle redirect)

NO REPORT IS GENERATED. The answer goes straight to the user.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import AIMessage

from agents.classifier import QueryIntent
from agents.state import GraphState
from models.agent import MCPCallRecord
from prompts.chat_prompt import (
    CAPABILITIES_MESSAGE,
    CHAT_SYSTEM_PROMPT,
    GREETING_MESSAGE,
    THANKS_MESSAGE,
    build_chat_prompt,
)
from services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# Patterns that should trigger a web search for factual grounding.
_FACTUAL_QUESTION_RE = (
    r"\b(what is|what are|who is|when did|when was|where is|how to|how much|how many|why|"
    r"meaning of|definition of|history of|difference between|news|latest|current|2025|2026|"
    r"weather|capital of|population of)\b"
)

# "Who is <person>?" / Hinglish "<name> kon/kaun hai" — identity lookups
# about a specific person. This agent's job is job-market intelligence,
# not a people-search tool: answering these via web search + LLM summary
# risks confidently discussing (or hallucinating about) a real, possibly
# private individual by name — a real observed case had it summarizing
# what it found about a specific named person from a resume filename.
# These get redirected instead of answered, regardless of who the person
# is (a public figure included) — that's simply out of this agent's scope.
_PERSON_LOOKUP_RE = re.compile(
    r"(\bwho\s+is\b|\bwho'?s\b|\bkon\s+hai\b|\bkaun\s+hai\b)", re.IGNORECASE
)

PERSON_LOOKUP_REDIRECT_MESSAGE = (
    "I'm scoped to tech job-market intelligence, not looking people up — I don't have a "
    "reliable way to verify who someone is, and getting that wrong about a real person is "
    "worse than not answering. Ask me about skills, salaries, tech trends, or how your "
    "resume matches the market instead!"
)


def _quick_template(intent: Optional[str], query: str) -> Optional[str]:
    """
    Return a fast template response for greetings/thanks/capabilities.

    Args:
        intent: The classified query intent.
        query: The raw user query (used for refined matching).

    Returns:
        Template string, or None if no template applies.
    """
    q = query.strip().lower()
    if intent == QueryIntent.GENERAL_CHAT.value:
        greeting_markers = ("hi", "hello", "hey", "good morning", "good afternoon",
                            "good evening", "hii", "yo", "namaste")
        if q.startswith(greeting_markers) and len(q) < 30:
            return GREETING_MESSAGE

        if q in ("thanks", "thank you", "thankyou", "thx", "ty"):
            return THANKS_MESSAGE

        capability_markers = ("what can you do", "what do you do", "how can you help",
                              "who are you", "your capabilities", "your features",
                              "what are you", "how do you work")
        # Bare "help"/"help me" is only a capability question when it's the
        # WHOLE message — "help" as a substring anywhere (e.g. "help me
        # compare salaries") is an ordinary request, not a request to
        # explain what this bot can do. This mirrors the same fix in
        # agents/classifier.py's _CAPABILITY_RE / _BARE_HELP_RE.
        is_bare_help = q.rstrip("!.? ") in ("help", "help me")
        if any(m in q for m in capability_markers) or is_bare_help:
            return CAPABILITIES_MESSAGE
    return None


def _wants_search(query: str) -> bool:
    """Decide whether the chat responder should ground with web search."""
    return re.search(_FACTUAL_QUESTION_RE, query, re.IGNORECASE) is not None


def _summarize_search_results(results: list[dict], max_results: int = 4) -> str:
    """Format web search results into a compact grounding block."""
    lines = []
    for item in results[:max_results]:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = (item.get("content") or "")[:300]
        lines.append(f"- **{title}** ({url})\n  {content}")
    return "\n".join(lines)


async def run_chat_responder(state: GraphState) -> dict:
    """
    ChatResponder agent node function.

    Produces a direct answer and stores it in state.direct_answer.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with direct_answer and messages.
    """
    query = state["user_query"]
    intent = state.get("query_intent") or QueryIntent.GENERAL_CHAT.value
    mcp_calls = state.get("mcp_calls", []) or []

    logger.info("ChatResponder starting", extra={"intent": intent, "query": query})

    # --- 1. Fast template responses (no LLM, no MCP) ------------------
    template = _quick_template(intent, query)
    if template:
        return {
            "direct_answer": template,
            "completed_agents": state.get("completed_agents", []) + ["chat_responder"],
            "messages": [AIMessage(content=template)],
        }

    # --- 1b. Person-identity lookups are out of scope, always -----------
    # Checked before web-search grounding so a "who is <name>" question
    # never reaches the search+LLM path at all, regardless of intent.
    if _PERSON_LOOKUP_RE.search(query):
        return {
            "direct_answer": PERSON_LOOKUP_REDIRECT_MESSAGE,
            "completed_agents": state.get("completed_agents", []) + ["chat_responder"],
            "messages": [AIMessage(content=PERSON_LOOKUP_REDIRECT_MESSAGE)],
        }

    # --- 2. Optional web-search grounding via search MCP ----------------
    search_context = ""
    search_call: Optional[MCPCallRecord] = None
    if _wants_search(query):
        try:
            from mcp_servers.registry import get_registry
            registry = get_registry()
            result = await registry.call_tool(
                "search",
                "web_search",
                {"query": query, "max_results": 5},
            )
            if result.success and result.result:
                search_call = MCPCallRecord(
                    agent="chat_responder",
                    server="search",
                    tool="web_search",
                    arguments={"query": query, "max_results": 5},
                    success=True,
                    summary=f"Fetched {len(result.result.get('results', []))} web results",
                )
                mcp_calls = [*mcp_calls, search_call.model_dump()]
                search_context = _summarize_search_results(result.result.get("results", []))
                logger.info("ChatResponder grounded answer with web search")
            else:
                logger.info(
                    "Web search unavailable for chat answer",
                    extra={"error": result.error},
                )
        except Exception as e:
            logger.warning("ChatResponder search MCP failed", extra={"error": str(e)})

    # --- 3. LLM-composed conversational answer -------------------------
    answer = ""
    try:
        llm = get_llm_service()
        answer = await llm.generate(
            user_prompt=build_chat_prompt(
                query,
                search_context,
                conversation_history=state.get("conversation_history") or [],
            ),
            system_prompt=CHAT_SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("ChatResponder LLM failed", extra={"error": str(e)})
        answer = (
            f"I'd love to answer that for you, but I'm having trouble reaching my "
            f"language model right now. In the meantime — I'm a **job market "
            f"intelligence agent** who analyzes real-time job postings for skills, "
            f"salaries, tech trends and resume fit. Want to try one of those?"
        )

    return {
        "direct_answer": answer.strip(),
        "mcp_calls": mcp_calls,
        "completed_agents": state.get("completed_agents", []) + ["chat_responder"],
        "messages": [AIMessage(content=answer.strip())],
    }
