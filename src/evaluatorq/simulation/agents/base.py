"""Base agent class for simulation agents.

Provides common functionality for all agents in the simulation system,
including LLM interaction with retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from evaluatorq.simulation.types import DEFAULT_MODEL, ChatMessage, TokenUsage
from evaluatorq.simulation.utils.retry import with_retry

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60


@dataclass
class LLMResult:
    """Result of a single LLM call, including optional tool calls."""

    content: str
    tool_calls: list[Any] | None = None


@dataclass
class AgentConfig:
    """Configuration options for constructing an agent."""

    model: str = DEFAULT_MODEL
    client: AsyncOpenAI | None = None
    api_key: str | None = None


class BaseAgent(ABC):
    """Abstract base class for simulation agents.

    Provides common LLM interaction functionality with exponential-backoff
    retry logic and cumulative token-usage tracking.

    **Client injection**: pass an existing ``AsyncOpenAI`` client via
    ``config.client`` to share a single HTTP connection across multiple agents.
    The agent will NOT close an injected client.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        config = config or AgentConfig()
        self._model = config.model

        if config.client is not None:
            self._client = config.client
            self._client_owned = False
        else:
            resolved_api_key = config.api_key or os.environ.get("ORQ_API_KEY")
            if not resolved_api_key:
                raise ValueError(
                    "ORQ_API_KEY environment variable is not set. Set it or pass api_key in AgentConfig."
                )
            self._client = AsyncOpenAI(
                base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
                api_key=resolved_api_key,
            )
            self._client_owned = True

        self._usage = TokenUsage()

    # ---------------------------------------------------------------------------
    # Abstract interface
    # ---------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for identification."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this agent."""

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    async def respond_async(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Generate a text response for a conversation."""
        result = await self._call_llm(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not result.content:
            raise RuntimeError(
                f"{self.name}: LLM call failed -- no content in response"
            )
        return result.content

    def get_usage(self) -> TokenUsage:
        """Get cumulative token usage for this agent."""
        return self._usage.model_copy()

    def reset_usage(self) -> None:
        """Reset token usage counters to zero."""
        self._usage = TokenUsage()

    async def close(self) -> None:
        """Close the underlying HTTP client (only if agent-owned)."""
        if self._client_owned and hasattr(self._client, "close"):
            await self._client.close()

    # ---------------------------------------------------------------------------
    # Protected helpers
    # ---------------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """Call the LLM with retry logic (exponential backoff).

        Retries on rate-limit (429) and server errors (500+). All other errors
        are raised immediately. ``asyncio.TimeoutError`` is never retried.
        """
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens or 2048
        timeout_s = timeout or DEFAULT_TIMEOUT_S

        full_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m.role, "content": m.content} for m in messages],
        ]

        params: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        async def _do_call() -> LLMResult:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(**params),
                timeout=timeout_s,
            )

            choice = response.choices[0] if response.choices else None
            if not choice:
                raise RuntimeError(f"{self.name}: No choices in response")

            message = choice.message

            # Accumulate token usage
            if response.usage:
                self._usage.prompt_tokens += response.usage.prompt_tokens
                self._usage.completion_tokens += response.usage.completion_tokens
                self._usage.total_tokens += response.usage.total_tokens

            result = LLMResult(content=message.content or "")
            if message.tool_calls:
                result.tool_calls = list(message.tool_calls)

            return result

        return await with_retry(_do_call, label=f"{self.name}._call_llm")
