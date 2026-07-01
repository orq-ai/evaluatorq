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
from typing import TYPE_CHECKING, Any

from openai import BadRequestError

from evaluatorq.common.llm_call import execute_chat_completion
from evaluatorq.common.retry import with_retry
from evaluatorq.common.tracing import get_trace_context_headers, record_llm_input, record_llm_response
from evaluatorq.contracts import (
    AgentResponse,
    FunctionCall,
    LLMCallConfig,
    StrategyToolCall,
    TextOutputItem,
    TokenUsage,
    ToolCallOutputItem,
    content_to_text,
)
from evaluatorq.openresponses.client import build_simulation_client
from evaluatorq.simulation.tracing import with_llm_span
from evaluatorq.simulation.types import DEFAULT_MODEL, Message

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f'Environment variable {name}={raw!r} must be a number') from None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f'Environment variable {name}={raw!r} must be an integer') from None


# Per-LLM-call timeout. Self-hosted endpoints (e.g. a single-GPU tailscale box
# under parallel load) can exceed the default; raise via EVALUATORQ_LLM_TIMEOUT_S.
DEFAULT_TIMEOUT_S = _env_float('EVALUATORQ_LLM_TIMEOUT_S', 60.0)

# Default completion-token budget. Reasoning models (e.g. gemma-4) spend tokens
# on hidden reasoning before the tool call; too small a budget truncates the
# response (finish_reason=length) before the tool call is emitted, surfacing as
# "no text and no tool calls". Raise via EVALUATORQ_LLM_MAX_TOKENS for such models.
DEFAULT_MAX_TOKENS = _env_int('EVALUATORQ_LLM_MAX_TOKENS', 8192)

# Default reasoning effort for reasoning-capable models. "medium" keeps hidden
# reasoning bounded (far fewer tokens than the model's default), which avoids
# budget-exhaustion truncation. Endpoints that don't support it degrade
# gracefully (the param is dropped on a 400). Set "" / "none" to omit it.
_REASONING_EFFORT_RAW = os.environ.get('EVALUATORQ_REASONING_EFFORT', 'medium').strip().lower()
DEFAULT_REASONING_EFFORT = _REASONING_EFFORT_RAW if _REASONING_EFFORT_RAW not in ('', 'none', 'off') else None


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
        messages: list[Message],
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
            raise RuntimeError(f'{self.name}: LLM call failed -- no content in response')
        return result.content

    def get_usage(self) -> TokenUsage:
        """Get cumulative token usage for this agent."""
        return self._usage.model_copy()

    def reset_usage(self) -> None:
        """Reset token usage counters to zero."""
        self._usage = TokenUsage()

    async def close(self) -> None:
        """Close the underlying HTTP client (only if agent-owned)."""
        if self._client_owned and hasattr(self._client, 'close'):
            await self._client.close()

    # ---------------------------------------------------------------------------
    # Protected helpers
    # ---------------------------------------------------------------------------

    def _build_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Construct (or reuse) an ``AsyncOpenAI`` client from ``self.config``.

        Delegates to :func:`evaluatorq.openresponses.client.build_simulation_client`.

        Resolution order:
        1. ``self.config.client`` — injected client, used as-is (not owned).
        2. ``api_key`` argument (extracted from legacy ``AgentConfig.api_key``),
           treated as an ORQ key and routed through the Orq router.
        3. ``ORQ_API_KEY`` env var — routes through
           ``ORQ_BASE_URL/v3/router`` (default: ``https://my.orq.ai/v3/router``).
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
        messages: list[Message],
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
        if self.config.api == 'responses':
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
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_purpose: str | None = None,
    ) -> LLMResult:
        """Call the LLM via the Chat Completions API with retry logic."""
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens or DEFAULT_MAX_TOKENS
        timeout_s = timeout or DEFAULT_TIMEOUT_S

        full_messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': self.system_prompt},
            *[{'role': m.role, 'content': content_to_text(m.content)} for m in messages],
        ]

        async with with_llm_span(
            model=self._model,
            operation='chat',
            temperature=temp,
            max_tokens=max_tok,
            purpose=llm_purpose,
        ) as span:
            reasoning_kwargs = {'reasoning_effort': DEFAULT_REASONING_EFFORT} if DEFAULT_REASONING_EFFORT else None

            async def _do_call() -> LLMResult:
                finish_reason: str | None = None
                for attempt in range(2):
                    response, delta = await execute_chat_completion(
                        client=self._client,
                        model=self._model,
                        messages=full_messages,
                        span=span,
                        timeout_s=timeout_s,
                        temperature=temp,
                        max_tokens=max_tok,
                        tools=tools,
                        extra_kwargs=reasoning_kwargs,
                    )
                    if delta is not None:
                        self._usage = self._usage + delta

                    choice = response.choices[0] if response.choices else None
                    if not choice:
                        raise RuntimeError(f'{self.name}: No choices in response')
                    message = choice.message
                    content = message.content
                    tool_calls = list(message.tool_calls or [])
                    if content or tool_calls:
                        return LLMResult(content=content or '', tool_calls=tool_calls or None)
                    finish_reason = choice.finish_reason
                    if attempt == 0:
                        logger.info(
                            '%s._call_chat_completions: empty response (finish_reason=%s), retrying once',
                            self.name,
                            finish_reason,
                        )
                # Truncated before the model could emit text/tool call. Common with
                # reasoning models whose hidden reasoning exhausts the token budget.
                if finish_reason == 'length':
                    raise RuntimeError(
                        f'{self.name}._call_chat_completions: response truncated (finish_reason=length, '
                        f'max_tokens={max_tok}) before any text or tool call. The model — likely a reasoning '
                        'model — ran out of tokens during reasoning. Raise the budget via EVALUATORQ_LLM_MAX_TOKENS.'
                    )
                raise RuntimeError(
                    f'{self.name}._call_chat_completions: LLM returned no text and no tool calls after retry '
                    f'(finish_reason={finish_reason}). Check model and prompt.'
                )

            return await with_retry(_do_call, label=f'{self.name}._call_chat_completions')

    async def _call_responses(
        self,
        messages: list[Message],
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

        input_messages = [{'role': m.role, 'content': content_to_text(m.content)} for m in messages]

        params: dict[str, Any] = {
            'model': self._model,
            'input': input_messages,
            'instructions': self.system_prompt,
        }

        if tools:
            params['tools'] = [_responses_tool_schema(tool) for tool in tools]

        if temperature is not None:
            params['temperature'] = temperature

        params['max_output_tokens'] = max_tokens or DEFAULT_MAX_TOKENS

        if DEFAULT_REASONING_EFFORT:
            params['reasoning'] = {'effort': DEFAULT_REASONING_EFFORT}

        async with with_llm_span(
            model=self._model,
            operation='responses',
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=llm_purpose,
        ) as span:
            # Responses API sends system context via `instructions`, not as a
            # message in `input`. Record what is actually sent so the span
            # matches the real request shape (mirrors _call_chat_completions
            # which records full_messages including the system entry).
            if span is not None:
                span.set_attribute('gen_ai.request.instructions', self.system_prompt[:2000])
            record_llm_input(span, input_messages)
            trace_headers = await get_trace_context_headers()

            async def _do_call() -> LLMResult:
                call_kwargs: dict[str, Any] = dict(params)
                if trace_headers:
                    call_kwargs['extra_headers'] = trace_headers
                try:
                    response = await asyncio.wait_for(
                        self._client.responses.create(**call_kwargs),
                        timeout=timeout_s,
                    )
                except BadRequestError as exc:
                    # "where possible": drop reasoning and retry if the endpoint
                    # rejects it, rather than failing the call. Gate on the error
                    # body so an unrelated 400 isn't masked by a stripped retry.
                    err_body = str(getattr(exc, 'body', None) or getattr(exc, 'message', '') or '').lower()
                    if 'reasoning' not in call_kwargs or 'reasoning' not in err_body:
                        raise
                    logger.warning(
                        '%s._call_responses: model %s rejected reasoning; dropping it and retrying once',
                        self.name,
                        self.config.model,
                    )
                    call_kwargs.pop('reasoning', None)
                    response = await asyncio.wait_for(
                        self._client.responses.create(**call_kwargs),
                        timeout=timeout_s,
                    )

                agent_response = AgentResponse.from_openresponses(response)
                output_items = agent_response.output
                usage = agent_response.usage

                record_llm_response(span, response)

                # Accumulate token usage (from_openresponses leaves calls=0, add 1)
                if usage is not None:
                    self._usage = self._usage + usage.model_copy(update={'calls': 1})

                # Separate text from tool-call items; isinstance guards prevent
                # ReasoningOutputItem.text leaking into response content.
                text_items = [i for i in output_items if isinstance(i, TextOutputItem)]
                tool_call_items = [i for i in output_items if isinstance(i, ToolCallOutputItem)]

                if not text_items and not tool_call_items:
                    # No text, no tool calls — warn but don't raise (redteam callers
                    # may handle empty). Surface the reason so it's clear to the user:
                    # reason=max_output_tokens means the budget was too small.
                    incomplete = getattr(response, 'incomplete_details', None)
                    reason = getattr(incomplete, 'reason', None) if incomplete else getattr(response, 'status', None)
                    logger.warning(
                        '%s._call_responses: empty response — no text or tool calls (model=%s, reason=%s). '
                        'If reason=max_output_tokens, raise the budget via EVALUATORQ_LLM_MAX_TOKENS.',
                        self.name,
                        self.config.model,
                        reason,
                    )

                text = ''.join(getattr(i, 'text', '') for i in text_items)
                result = LLMResult(content=text)
                if tool_call_items:
                    result.tool_calls = [
                        StrategyToolCall(
                            id=item.call_id,
                            function=FunctionCall(name=item.name, arguments=item.arguments),
                        )
                        for item in tool_call_items
                    ]
                return result

            return await with_retry(_do_call, label=f'{self.name}._call_responses')


def _responses_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert chat-completions function tools to Responses function tools."""
    if tool.get('type') == 'function' and isinstance(tool.get('function'), dict):
        fn = tool['function']
        return {
            'type': 'function',
            'name': fn.get('name'),
            'description': fn.get('description'),
            'parameters': fn.get('parameters') or {'type': 'object', 'properties': {}},
        }
    return tool
