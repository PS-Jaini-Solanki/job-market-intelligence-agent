"""
services/llm_service.py
========================
Groq LLM client wrapper with retry logic and structured output.

This service is the ONLY place in the codebase that talks to the LLM.
All agents use this service — they never create LLM clients directly.

Key features:
1. Retry logic with exponential backoff (tenacity library)
2. Structured output using Pydantic models
3. Token usage tracking
4. Consistent error handling
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings

logger = logging.getLogger(__name__)

# TypeVar for structured output — T must be a Pydantic BaseModel
T = TypeVar("T", bound=BaseModel)


class LLMService:
    """
    Wrapper around the Groq LLM client.

    Usage:
        llm_service = LLMService()

        # Simple text response
        response = await llm_service.generate("What is Python?")

        # Structured output (Pydantic model)
        result = await llm_service.generate_structured(
            prompt="Extract skills from: ...",
            output_model=SkillList,
        )
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Optional[ChatGroq] = None

    @property
    def client(self) -> ChatGroq:
        """
        Lazy-initialize the ChatGroq client.

        We create it on first use so the constructor doesn't
        fail if the API key isn't set yet (e.g., during testing).
        """
        if self._client is None:
            self._client = ChatGroq(
                api_key=self._settings.GROQ_API_KEY,
                model=self._settings.GROQ_MODEL,
                max_tokens=self._settings.GROQ_MAX_TOKENS,
                temperature=self._settings.GROQ_TEMPERATURE,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a text response from the LLM.

        Retries up to 3 times on failure with exponential backoff.

        Args:
            user_prompt: The user message to send.
            system_prompt: Optional system context to prepend.

        Returns:
            The LLM's text response.
        """
        messages: list[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        logger.debug(
            "Calling LLM",
            extra={
                "model": self._settings.GROQ_MODEL,
                "prompt_length": len(user_prompt),
            },
        )

        response = await self.client.ainvoke(messages)
        content = str(response.content)

        logger.debug(
            "LLM response received",
            extra={"response_length": len(content)},
        )

        return content

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def generate_structured(
        self,
        user_prompt: str,
        output_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Generate a structured response validated against a Pydantic model.

        The LLM is instructed to respond in JSON format matching the
        output_model schema. The response is then validated by Pydantic.

        Args:
            user_prompt: The user message to send.
            output_model: Pydantic model class to validate the response against.
            system_prompt: Optional system context.

        Returns:
            A validated instance of output_model.

        Raises:
            ValueError: If the LLM response cannot be parsed as the model.
        """
        # Use LangChain's with_structured_output for reliable JSON output
        structured_client = self.client.with_structured_output(
            output_model,
            method="json_mode",
        )

        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        logger.debug(
            "Calling LLM for structured output",
            extra={"output_model": output_model.__name__},
        )

        result = await structured_client.ainvoke(messages)

        logger.debug(
            "Structured output received",
            extra={"model_type": output_model.__name__},
        )

        return result  # type: ignore[return-value]

    async def generate_with_tools(
        self,
        user_prompt: str,
        tools: list[Any],
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        Call the LLM with tools bound to it (tool calling mode).

        The LLM can decide to call any of the provided tools.
        This is used by the planner agent.

        Args:
            user_prompt: The user message.
            tools: List of LangChain tool objects.
            system_prompt: Optional system context.

        Returns:
            AIMessage that may contain tool_calls.
        """
        client_with_tools = self.client.bind_tools(tools)

        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        logger.debug("Calling LLM with tools", extra={"tool_count": len(tools)})
        return await client_with_tools.ainvoke(messages)


# Module-level singleton — all code should use this instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Return the shared LLM service instance.

    Uses a module-level singleton to avoid creating multiple clients.
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
