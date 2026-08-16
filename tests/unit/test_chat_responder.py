"""Unit tests for the ChatResponder agent (agents/chat_responder.py)."""

import asyncio

from agents.chat_responder import (
    PERSON_LOOKUP_REDIRECT_MESSAGE,
    _quick_template,
    _wants_search,
    run_chat_responder,
)
from agents.classifier import QueryIntent
from agents.state import create_initial_state


def _run(query: str, intent: str = QueryIntent.GENERAL_CHAT.value) -> str:
    state = create_initial_state(query)
    state["query_intent"] = intent
    result = asyncio.run(run_chat_responder(state))
    return result["direct_answer"]


def test_person_lookup_who_is_redirected():
    # Regression: "who is <name>" questions about real people (a real
    # observed case involved a named individual whose resume happened to
    # be on disk) were fully answered via web search + LLM summary. This
    # agent is scoped to job-market intelligence, not people lookups.
    assert _run("who is Jenil CAPTAIN") == PERSON_LOOKUP_REDIRECT_MESSAGE


def test_person_lookup_hinglish_kon_hai_redirected():
    assert _run("Narendra bhai kon hai?") == PERSON_LOOKUP_REDIRECT_MESSAGE


def test_person_lookup_redirect_applies_even_to_public_figures():
    assert _run("who is the CEO of OpenAI") == PERSON_LOOKUP_REDIRECT_MESSAGE


def test_factual_definition_question_not_treated_as_person_lookup():
    # "what is an API" must still be answerable (not misclassified as a
    # person lookup just because it's a factual question).
    assert _wants_search("can you explain me what is api") is True


def test_bare_help_still_returns_capabilities_template():
    assert _quick_template(QueryIntent.GENERAL_CHAT.value, "help me") is not None
    assert _quick_template(QueryIntent.GENERAL_CHAT.value, "help") is not None


def test_help_me_with_real_request_is_not_a_capability_question():
    # Regression: "help me <do something specific>" was hijacked into the
    # generic capabilities blurb just because it contains "help" — this
    # checks the chat_responder's own (redundant) capability check, which
    # mirrors the same fix in agents/classifier.py's _CAPABILITY_RE.
    template = _quick_template(
        QueryIntent.GENERAL_CHAT.value,
        "help me provide a breakdown of job markets in India",
    )
    assert template is None
