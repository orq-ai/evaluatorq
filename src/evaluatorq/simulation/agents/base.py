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

from evaluatorq.contracts import LLMCallConfig
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
    """Configuration options for constructing an agent.

    .. deprecated::
        Use :class:`evaluatorq.contracts.LLMCallConfig` instead.
        ``AgentConfig`` is kept for backwards compatibility and will be
        removed in a future release.  Subclasses (``JudgeAgentConfig``,
        ``UserSimulatorAgentConfig``) will be migrated in a subsequent task.
    """

    model: str = DEFAULT_MODEL
    client: AsyncOpenAI | None = None
    api_key: str | None = None


def _config_from_agent_config(agent_cfg: AgentConfig) -> tuple[LLMCallConfig, str | None]:
    """Convert a legacy AgentConfig into a LLMCallConfig + optional api_key."""
    return (
        LLMCallConfig(
            model=agent_cfg.model,
            client=agent_cfg.client,
        ),
        agent_cfg.api_key,
    )


class BaseAgent(ABC):
    """Abstract base class for simulation agents.

    Provides common LLM interaction functionality with exponential-backoff
    retry logic and cumulative token-usage tracking.

    **Client injection**: pass an existing ``AsyncOpenAI`` client via
    ``config.client`` to share a single HTTP connection across multiple agents.
    The agent will NOT close an injected client.
    """

    def __init__(self, config: LLMCallConfig | AgentConfig | None = None) -> None:
        # Normalise legacy AgentConfig into LLMCallConfig
        extra_api_key: str | None = None
        if isinstance(config, AgentConfig):
            self.config, extra_api_key = _config_from_agent_config(config)
        else:
            self.config = config or LLMCallConfig(model=DEFAULT_MODEL)

        self._client_owned: bool
        self._client: AsyncOpenAI = self._build_client(extra_api_key)
        self._model = self.config.model
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

    def _build_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Construct (or reuse) an ``AsyncOpenAI`` client from ``self.config``.

        Resolution order for the API key:
        1. ``self.config.client`` — injected client, used as-is (not owned).
        2. ``api_key`` argument (extracted from legacy ``AgentConfig.api_key``).
        3. ``ORQ_API_KEY`` environment variable.
        4. ``OPENAI_API_KEY`` environment variable.

        The base URL defaults to ``ORQ_BASE_URL/v2/router`` (or
        ``https://api.orq.ai/v2/router``) when no injected client is provided.
        """
        if self.config.client is not None:
            self._client_owned = False
            return self.config.client  # type: ignore[return-value]

        resolved_api_key = (
            api_key
            or os.environ.get("ORQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "No API key found. Set ORQ_API_KEY or OPENAI_API_KEY, "
                "or pass api_key in AgentConfig / a pre-built client in LLMCallConfig."
            )

        base_url = (
            f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router"
        )

        self._client_owned = True
        return AsyncOpenAI(base_url=base_url, api_key=resolved_api_key)

    async def _call_llm(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """Call the LLM with retry logic, dispatching to chat or responses API.

        Retries on rate-limit (429) and server errors (500+). All other errors
        are raised immediately. ``asyncio.TimeoutError`` is never retried.
        """
        if self.config.api == "responses":
            return await self._call_responses(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                tools=tools,
            )
        return await self._call_chat_completions(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tools,
        )

    async def _call_chat_completions(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """Call the LLM via the Chat Completions API with retry logic."""
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

    async def _call_responses(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """Call the LLM via the OpenAI Responses API with retry logic.

        The Responses API uses ``input`` (list of message dicts) and
        ``instructions`` (system prompt) instead of a ``messages`` list.
        Text is extracted from ``response.output`` items that carry a
        ``content`` list of parts with a ``text`` attribute.
        """
        timeout_s = timeout or DEFAULT_TIMEOUT_S

        input_messages = [{"role": m.role, "content": m.content} for m in messages]

        params: dict[str, Any] = {
            "model": self._model,
            "input": input_messages,
            "instructions": self.system_prompt,
        }

        if tools:
            params["tools"] = tools

        if temperature is not None:
            params["temperature"] = temperature

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        async def _do_call() -> LLMResult:
            response = await asyncio.wait_for(
                self._client.responses.create(**params),
                timeout=timeout_s,
            )

            # Extract text from output items
            text = ""
            tool_calls: list[Any] = []
            for item in response.output or []:
                # Text message output items carry a ``content`` list
                if hasattr(item, "content"):
                    for part in item.content or []:
                        if hasattr(part, "text"):
                            text += part.text
                # Tool-call output items expose ``name`` / ``arguments``
                if hasattr(item, "name") and hasattr(item, "arguments"):
                    tool_calls.append(item)

            # Accumulate token usage if the SDK exposes it
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._usage.prompt_tokens += getattr(usage, "input_tokens", 0)
                self._usage.completion_tokens += getattr(usage, "output_tokens", 0)
                self._usage.total_tokens += getattr(usage, "total_tokens", 0)

            result = LLMResult(content=text)
            if tool_calls:
                result.tool_calls = tool_calls
            return result

        return await with_retry(_do_call, label=f"{self.name}._call_responses")
