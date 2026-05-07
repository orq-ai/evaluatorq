"""Base agent class for simulation agents.

Provides common functionality for all agents in the simulation system,
including LLM interaction with retry logic.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation._client import build_simulation_client, extract_responses_output
from evaluatorq.simulation.tracing import (
    get_trace_context_headers,
    record_llm_input,
    record_llm_response,
    with_llm_span,
)
from evaluatorq.simulation.types import DEFAULT_MODEL, ChatMessage, TokenUsage
from evaluatorq.simulation.utils.retry import with_retry

if TYPE_CHECKING:
    from openai import AsyncOpenAI

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
        llm_purpose: str | None = None,
    ) -> str:
        """Generate a text response for a conversation."""
        result = await self._call_llm(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            llm_purpose=llm_purpose,
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

        Delegates to :func:`evaluatorq.simulation._client.build_simulation_client`.

        Resolution order:
        1. ``self.config.client`` — injected client, used as-is (not owned).
        2. ``api_key`` argument (extracted from legacy ``AgentConfig.api_key``),
           treated as an ORQ key and routed through the Orq router.
        3. ``ORQ_API_KEY`` env var — routes through
           ``ORQ_BASE_URL/v2/router`` (default: ``https://api.orq.ai/v2/router``).
        4. ``OPENAI_API_KEY`` env var — uses the OpenAI SDK default base URL so
           traffic goes to OpenAI directly, not to the Orq router.
        """
        client, owned = build_simulation_client(
            self.config.client,
            extra_api_key=api_key,
        )
        self._client_owned = owned
        return client

    async def _call_llm(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_purpose: str | None = None,
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
                llm_purpose=llm_purpose,
            )
        return await self._call_chat_completions(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tools,
            llm_purpose=llm_purpose,
        )

    async def _call_chat_completions(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_purpose: str | None = None,
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

        async with with_llm_span(
            model=self._model,
            operation="chat",
            temperature=temp,
            max_tokens=max_tok,
            purpose=llm_purpose,
        ) as span:
            record_llm_input(
                span,
                [{"role": str(m["role"]), "content": str(m["content"])} for m in full_messages],
            )

            # Inject W3C trace context so the router can link its spans to
            # the current simulation trace. Active span and trace context
            # don't change across retries — compute headers once.
            trace_headers = await get_trace_context_headers()

            async def _do_call() -> LLMResult:
                call_kwargs: dict[str, Any] = dict(params)
                if trace_headers:
                    call_kwargs["extra_headers"] = trace_headers
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(**call_kwargs),
                    timeout=timeout_s,
                )

                choice = response.choices[0] if response.choices else None
                if not choice:
                    raise RuntimeError(f"{self.name}: No choices in response")

                message = choice.message

                # Record LLM response on the span (token usage, finish reason, etc.)
                record_llm_response(span, response)

                # Accumulate token usage
                if response.usage:
                    self._usage.prompt_tokens += response.usage.prompt_tokens
                    self._usage.completion_tokens += response.usage.completion_tokens
                    self._usage.total_tokens += response.usage.total_tokens

                content = message.content
                tool_calls = list(message.tool_calls or [])
                if not content and not tool_calls:
                    raise RuntimeError(
                        f"{self.name}._call_chat_completions: LLM returned no text and no tool calls. "
                        "Check model and prompt."
                    )
                return LLMResult(content=content or "", tool_calls=tool_calls or None)

            return await with_retry(_do_call, label=f"{self.name}._call_chat_completions")

    async def _call_responses(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_purpose: str | None = None,
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

        async with with_llm_span(
            model=self._model,
            operation="responses",
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=llm_purpose,
        ) as span:
            record_llm_input(
                span,
                [{"role": "system", "content": self.system_prompt}, *input_messages],
            )
            trace_headers = await get_trace_context_headers()

            async def _do_call() -> LLMResult:
                call_kwargs: dict[str, Any] = dict(params)
                if trace_headers:
                    call_kwargs["extra_headers"] = trace_headers
                response = await asyncio.wait_for(
                    self._client.responses.create(**call_kwargs),
                    timeout=timeout_s,
                )

                output_items, usage = extract_responses_output(response)

                record_llm_response(span, response)

                # Accumulate token usage
                self._usage.prompt_tokens += usage.prompt_tokens
                self._usage.completion_tokens += usage.completion_tokens
                self._usage.total_tokens += usage.total_tokens

                # Separate text from tool-call items
                text_items = [i for i in output_items if hasattr(i, "text")]
                tool_call_items = [i for i in output_items if hasattr(i, "call_id")]

                if not text_items and not tool_call_items:
                    # No text, no tool calls — warn but don't raise (redteam callers may handle empty)
                    logger.warning(
                        "%s._call_responses: no text or tool calls in response (model=%s)",
                        self.name,
                        self.config.model,
                    )

                text = "".join(getattr(i, "text", "") for i in text_items)
                result = LLMResult(content=text)
                if tool_call_items:
                    result.tool_calls = tool_call_items
                return result

            return await with_retry(_do_call, label=f"{self.name}._call_responses")
