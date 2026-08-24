"""Base agent class for simulation agents.

Provides common functionality for all agents in the simulation system,
including LLM interaction with retry logic.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from evaluatorq.common.llm_call import (
    execute_chat_completion,
    execute_response,
)
from evaluatorq.common.llm_client import client_routes_through_orq
from evaluatorq.common.prompt_cache import (
    apply_cache_breakpoints,
    caching_applies,
    mark_responses_input,
    responses_volatile_items,
)
from evaluatorq.common.responses import first_responses_refusal, responses_stop_reason
from evaluatorq.common.retry import with_retry
from evaluatorq.common.thread_context import thread_body_param
from evaluatorq.common.tracing import record_llm_input
from evaluatorq.contracts import (
    DEFAULT_TARGET_MAX_TOKENS,
    AgentResponse,
    FunctionCall,
    LLMCallConfig,
    StrategyToolCall,
    TextOutputItem,
    ToolCallOutputItem,
)
from evaluatorq.openresponses.client import build_simulation_client
from evaluatorq.openresponses.input_items import messages_to_responses_input
from evaluatorq.simulation._usage import UsageTracking
from evaluatorq.simulation.tracing import span_message_text, with_llm_span
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


# The three functions below resolve at CALL TIME, and are the process-global
# fallback only: an explicitly set `LLMCallConfig` field always wins.


def _default_timeout_s() -> float:
    """Per-LLM-call timeout fallback. Self-hosted endpoints (e.g. a single-GPU
    tailscale box under parallel load) can exceed the default; raise via
    EVALUATORQ_LLM_TIMEOUT_S, or per-agent via ``LLMCallConfig.timeout_ms``.
    """
    return _env_float('EVALUATORQ_LLM_TIMEOUT_S', 60.0)


def _default_max_tokens() -> int:
    """Default completion-token budget fallback, shared with red team via
    DEFAULT_TARGET_MAX_TOKENS. Reasoning models (e.g. gemma-4) spend tokens on
    hidden reasoning before the tool call; too small a budget truncates the
    response (finish_reason=length) before the tool call is emitted, surfacing
    as "no text and no tool calls". Raise via EVALUATORQ_LLM_MAX_TOKENS, or
    per-agent via ``LLMCallConfig.max_tokens``.
    """
    return _env_int('EVALUATORQ_LLM_MAX_TOKENS', DEFAULT_TARGET_MAX_TOKENS)


def _default_reasoning_effort() -> str | None:
    """Reasoning effort from ``EVALUATORQ_REASONING_EFFORT``, or ``None`` when unset.

    There is deliberately no global default. Sending an effort the user did not ask
    for costs a rejected request plus a retry on every model that does not support
    the parameter, and it overrides the model's own tuned default on every model
    that does. Unset means "say nothing and let the model decide".

    Set ``LLMCallConfig.reasoning_effort`` per-agent to override the env value
    (including an explicit ``None`` to opt out). ``""`` / ``none`` / ``off`` in the
    env var also resolve to ``None``.

    A separate knob from red team's ``LLMConfig.target_reasoning_effort``, which
    configures the agent *under test* rather than the simulator's own LLM calls
    (user simulator, judge). Both ultimately resolve into a per-call
    ``LLMCallConfig.reasoning_effort`` — this one via the env fallback below when
    a `BaseAgent`'s config leaves it unset, that one via an explicit
    ``reasoning_effort=`` threaded into the target's backend construction.
    """
    raw = os.environ.get('EVALUATORQ_REASONING_EFFORT', '').strip().lower()
    return raw if raw not in ('', 'none', 'off') else None


# Backward-compat snapshot for external callers. No call path reads it — every call
# resolves `_default_max_tokens()` live — so it can go stale without effect.
DEFAULT_MAX_TOKENS = _default_max_tokens()


@dataclass
class LLMResult:
    """Result of a single LLM call, including optional tool calls."""

    content: str
    tool_calls: list[Any] | None = None
    refusal: str | None = None


@dataclass
class AgentConfig:
    """Configuration options for constructing an agent.

    Deprecated: use `evaluatorq.contracts.LLMCallConfig` instead. `AgentConfig`
    is kept for backwards compatibility and will be removed in a future release.

    ``temperature`` / ``max_tokens`` / ``timeout_ms`` / ``extra_kwargs`` /
    ``reasoning_effort`` / ``retry_count`` default to ``None`` here (not
    `LLMCallConfig`'s own defaults): ``None`` means "caller didn't touch this",
    so `_config_from_agent_config` omits it from the constructed
    `LLMCallConfig`, letting the per-call-site literal / env fallback apply —
    exactly as if this legacy class had never been in the way.
    """

    model: str = DEFAULT_MODEL
    client: AsyncOpenAI | None = None
    api_key: str | None = None
    api: Literal['chat_completions', 'responses'] = 'chat_completions'
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_ms: int | None = None
    extra_kwargs: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    retry_count: int | None = None

    @classmethod
    def from_call_config(cls, config: LLMCallConfig, *, api_key: str | None = None, **overrides: Any) -> Self:
        """Build this config class from a `LLMCallConfig`, preserving caller intent.

        Only fields the caller explicitly set on ``config`` are carried over —
        keyed on ``model_fields_set``, not on the values — so a round trip back
        through `_config_from_agent_config` reproduces the same
        ``model_fields_set`` and the `BaseAgent` resolvers still tell "caller set
        this" apart from "field default". Copying values instead would pin every
        `LLMCallConfig` default onto the agent and shadow the per-call-site
        literals, which is exactly what this class's docstring warns about.

        ``overrides`` go to the subclass's own fields (``system_prompt``,
        ``goal``, ...) and win over anything derived from ``config``.

        One field cannot survive the round trip: an explicitly set
        ``temperature=None``. ``AgentConfig`` spells "caller didn't touch this"
        as ``None``, so the two collapse and the call-site literal applies. Pass
        `LLMCallConfig` straight to the agent when you need that distinction.
        """
        kwargs: dict[str, Any] = {
            'model': config.model,
            'client': config.client,
            'api_key': api_key,
            **config.set_values(
                'api',
                'temperature',
                'max_tokens',
                'timeout_ms',
                'extra_kwargs',
                'extra_body',
                'reasoning_effort',
                'retry_count',
            ),
        }
        return cls(**{**kwargs, **overrides})


def _config_from_agent_config(agent_cfg: AgentConfig) -> tuple[LLMCallConfig, str | None]:
    """Convert a legacy AgentConfig into a LLMCallConfig + optional api_key.

    Only fields the caller actually set on ``agent_cfg`` (non-``None``) are
    passed through to the `LLMCallConfig` constructor, so
    `LLMCallConfig.model_fields_set` accurately reflects caller intent for the
    resolvers in `BaseAgent` (``_resolved_temperature`` etc.) — a field left at
    its `AgentConfig` default of ``None`` must NOT shadow the per-call-site
    literal / env fallback with `LLMCallConfig`'s own field default.
    """
    kwargs: dict[str, Any] = {'model': agent_cfg.model, 'client': agent_cfg.client, 'api': agent_cfg.api}
    if agent_cfg.temperature is not None:
        kwargs['temperature'] = agent_cfg.temperature
    if agent_cfg.max_tokens is not None:
        kwargs['max_tokens'] = agent_cfg.max_tokens
    if agent_cfg.timeout_ms is not None:
        kwargs['timeout_ms'] = agent_cfg.timeout_ms
    if agent_cfg.extra_kwargs is not None:
        kwargs['extra_kwargs'] = agent_cfg.extra_kwargs
    if agent_cfg.extra_body is not None:
        kwargs['extra_body'] = agent_cfg.extra_body
    if agent_cfg.reasoning_effort is not None:
        kwargs['reasoning_effort'] = agent_cfg.reasoning_effort
    if agent_cfg.retry_count is not None:
        kwargs['retry_count'] = agent_cfg.retry_count
    return LLMCallConfig(**kwargs), agent_cfg.api_key


class BaseAgent(UsageTracking, ABC):
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
        self.reset_usage()

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
        volatile_tail: int = 0,
    ) -> str:
        """Generate a text response for a conversation.

        ``volatile_tail`` is the number of trailing messages this caller rebuilds
        every turn instead of appending to the transcript — a per-call
        instruction, a re-rendered scratchpad. It keeps the prompt-cache
        breakpoint off them; see `common.prompt_cache`. It defaults to ``0``
        because most callers replay a transcript verbatim, but a caller that
        appends anything synthetic must say so or pay a per-turn cache write
        nothing reads back.
        """
        result = await self._call_llm(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            llm_purpose=llm_purpose,
            volatile_tail=volatile_tail,
        )
        if not result.content:
            raise RuntimeError(f'{self.name}: LLM call failed -- no content in response')
        return result.content

    async def close(self) -> None:
        """Close the underlying HTTP client (only if agent-owned)."""
        if self._client_owned and hasattr(self._client, 'close'):
            await self._client.close()

    # ---------------------------------------------------------------------------
    # Protected helpers
    # ---------------------------------------------------------------------------

    def _build_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Construct (or reuse) an ``AsyncOpenAI`` client from ``self.config``.

        Delegates to `evaluatorq.openresponses.client.build_simulation_client`.

        Resolution order:
        1. ``self.config.client`` — injected client, used as-is (not owned).
        2. ``api_key`` argument (extracted from legacy ``AgentConfig.api_key``),
           treated as an ORQ key and routed through the Orq router.
        3. ``ORQ_API_KEY`` env var — routes through
           ``ORQ_BASE_URL/v3/router`` (default: ``https://my.orq.ai/v3/router``).
        4. ``OPENAI_API_KEY`` env var — uses the OpenAI SDK default base URL so
           traffic goes to OpenAI directly, not to the Orq router.

        Retry owner: ``with_retry`` in ``_call_chat_completions`` / ``_call_responses``,
        bounded at ``config.retry_count + 1`` attempts. The SDK budget is disarmed
        here (``max_retries=0``) so the two cannot multiply.
        """
        client, owned = build_simulation_client(
            self.config.client,
            extra_api_key=api_key,
            max_retries=0,
        )
        self._client_owned = owned
        return client

    def _resolved_temperature(self, call_value: float | None) -> float | None:
        """Effective temperature: an explicitly set ``self.config.temperature``
        beats the per-call ``call_value`` — including an explicit ``None``, which
        opts this agent out of the call site's value on purpose. Unresolved means
        the request omits the parameter, which is what reasoning-class models
        need: they answer 400 to ``temperature`` at any value.

        Gated on ``model_fields_set`` rather than on the value, matching
        `_resolved_max_tokens` / `_resolved_timeout_s` / `_resolved_reasoning_effort`.
        `LLMCallConfig` now defaults ``temperature`` to ``None``, so a value check
        could not tell an explicit ``None`` from an untouched field.
        """
        if 'temperature' in self.config.model_fields_set:
            return self.config.temperature
        return call_value

    def _resolved_max_tokens(self, call_value: int | None) -> int:
        """Effective max-tokens budget: explicit ``self.config.max_tokens`` beats
        ``call_value``, which beats the call-time-resolved env fallback
        (`_default_max_tokens`, EVALUATORQ_LLM_MAX_TOKENS).

        Pair with `_max_tokens_advice` when building a truncation message: it
        derives from the same ``model_fields_set`` check, so the two can't drift.
        """
        if 'max_tokens' in self.config.model_fields_set:
            return self.config.max_tokens
        return call_value if call_value is not None else _default_max_tokens()

    def _max_tokens_advice(self, call_value: int | None) -> str:
        """Remedy text for a truncation message, naming whichever knob
        `_resolved_max_tokens` actually used for this agent.

        Pass the same ``call_value`` that was handed to `_resolved_max_tokens`,
        so the message walks the identical three tiers: config, then the
        caller's per-call ``max_tokens=``, then the env fallback.

        A user whose `LLMCallConfig.max_tokens` is pinned and raises
        ``EVALUATORQ_LLM_MAX_TOKENS`` instead sees no change and no signal why —
        the env var is only consulted when the config leaves ``max_tokens``
        unset *and* the caller passed nothing.
        """
        if 'max_tokens' in self.config.model_fields_set:
            return "raise max_tokens on this agent's LLMCallConfig"
        if call_value is not None:
            return 'raise the max_tokens argument passed to this call'
        return 'raise the budget via EVALUATORQ_LLM_MAX_TOKENS'

    def _resolved_timeout_s(self, call_value: float | None) -> float:
        """Effective per-call timeout in seconds: explicit ``self.config.timeout_ms``
        (converted from ms) beats ``call_value`` (already seconds — the unit
        `respond_async` / `_call_llm` use), which beats the call-time-resolved
        env fallback (`_default_timeout_s`, EVALUATORQ_LLM_TIMEOUT_S).
        """
        if 'timeout_ms' in self.config.model_fields_set:
            return self.config.timeout_ms / 1000.0
        return call_value if call_value is not None else _default_timeout_s()

    def _resolved_reasoning_effort(self) -> str | None:
        """Effective reasoning effort: an explicitly set ``self.config.reasoning_effort``
        wins — including an explicit ``None``, which opts this agent out of the
        env fallback on purpose — else the call-time-resolved env fallback
        (`_default_reasoning_effort`, EVALUATORQ_REASONING_EFFORT).

        The explicit-``None`` opt-out is reachable only by passing an
        ``LLMCallConfig`` directly. `_config_from_agent_config` forwards only
        non-``None`` fields, so on the legacy `AgentConfig` path
        ``reasoning_effort=None`` is indistinguishable from unset and the env
        fallback still applies.
        """
        if 'reasoning_effort' in self.config.model_fields_set:
            return self.config.reasoning_effort
        return _default_reasoning_effort()

    async def _call_llm(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_purpose: str | None = None,
        volatile_tail: int = 0,
    ) -> LLMResult:
        """Call the LLM with retry logic, dispatching to chat or responses API.

        Retries on rate-limit (429) and server errors (500+). All other errors
        are raised immediately. ``asyncio.TimeoutError`` is never retried. Retry
        is owned by ``with_retry``; client retries are disabled.

        ``volatile_tail`` is the number of trailing messages this caller rebuilds
        every turn instead of appending to the transcript. It keeps the cache
        breakpoint off them on both paths — see `common.prompt_cache`:
        `apply_cache_breakpoints` on the chat path, `mark_responses_input` on the
        Responses path (which is the judge's default).
        """
        if self.config.api == 'responses':
            return await self._call_responses(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                tools=tools,
                llm_purpose=llm_purpose,
                volatile_tail=volatile_tail,
            )
        return await self._call_chat_completions(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tools,
            llm_purpose=llm_purpose,
            volatile_tail=volatile_tail,
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
        volatile_tail: int = 0,
    ) -> LLMResult:
        """Call the LLM via the Chat Completions API with retry logic.

        Effective ``temperature`` / ``max_tokens`` / ``timeout`` / reasoning
        effort come from `_resolved_temperature` / `_resolved_max_tokens` /
        `_resolved_timeout_s` / `_resolved_reasoning_effort`: an explicit
        ``self.config`` value wins, else this call site's own argument, else the
        call-time env fallback. Temperature has no env tier — nothing supplies one,
        so an unresolved temperature is omitted from the request rather than
        defaulted. ``self.config.extra_kwargs`` rides along last, so it can still override
        any of the above (matches `LLMCallConfig.request_params`'s chat-completions contract).
        """
        temp = self._resolved_temperature(temperature)
        max_tok = self._resolved_max_tokens(max_tokens)
        timeout_s = self._resolved_timeout_s(timeout)
        reasoning_effort = self._resolved_reasoning_effort()

        full_messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': self.system_prompt},
            *[m.to_chat_completion() for m in messages],
        ]
        # Breakpoints on the system prompt and the end of the persisted transcript:
        # simulation replays a growing append-only prefix, so without them Anthropic
        # models re-encode the whole thing every turn (see common/prompt_cache.py).
        if caching_applies(self._client, self._model):
            full_messages = apply_cache_breakpoints(full_messages, volatile_tail=volatile_tail)

        async with with_llm_span(
            model=self._model,
            operation='chat',
            temperature=temp,
            max_tokens=max_tok,
            purpose=llm_purpose,
        ) as span:
            call_extra: dict[str, Any] = dict(self.config.extra_kwargs) if self.config.extra_kwargs else {}
            if reasoning_effort and 'reasoning_effort' not in call_extra:
                call_extra['reasoning_effort'] = reasoning_effort
            # execute_chat_completion treats None as "no extras"; an empty dict
            # would be splatted as a no-op but reads as "the caller set extras".
            reasoning_kwargs: dict[str, Any] | None = call_extra or None

            async def _do_call() -> LLMResult:
                finish_reason: str | None = None
                # Content-level retry inside one transport attempt: costs up to 2x the budget.
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
                        extra_body=self.config.extra_body or None,
                        extra_kwargs=reasoning_kwargs,
                    )
                    if delta is not None:
                        self._accumulate(delta)

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
                        f'model — ran out of tokens during reasoning; {self._max_tokens_advice(max_tokens)}.'
                    )
                raise RuntimeError(
                    f'{self.name}._call_chat_completions: LLM returned no text and no tool calls after retry '
                    f'(finish_reason={finish_reason}). Check model and prompt.'
                )

            return await with_retry(
                _do_call,
                label=f'{self.name}._call_chat_completions',
                max_attempts=self.config.retry_count + 1,
            )

    async def _call_responses(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_purpose: str | None = None,
        volatile_tail: int = 0,
    ) -> LLMResult:
        """Call the LLM via the OpenAI Responses API with retry logic.

        The Responses API uses ``input`` (list of message dicts) and
        ``instructions`` (system prompt) instead of a ``messages`` list.
        Text is extracted from ``response.output`` items that carry a
        ``content`` list of parts with a ``text`` attribute.

        Effective ``temperature`` / ``max_tokens`` / ``timeout`` / reasoning
        effort follow the same config-beats-call-site-beats-fallback order as
        `_call_chat_completions`, temperature's missing env tier included
        (see `_resolved_temperature` etc.).
        The request itself goes through `common.llm_call.execute_response`, so
        this leg gets the same slot limiting, reasoning-rejection memo, pipeline
        metadata, trace headers and `price_usage` call as every other Responses
        caller. `self.config.extra_kwargs` is merged in there last (structural
        keys guarded by `check_reserved_keys`), so it can override any of the
        above — `LLMCallConfig.request_params`'s Responses contract.
        """
        timeout_s = self._resolved_timeout_s(timeout)
        resolved_temp = self._resolved_temperature(temperature)
        max_tok = self._resolved_max_tokens(max_tokens)
        reasoning_effort = self._resolved_reasoning_effort()

        # Canonical renderer: an assistant turn must arrive as output_text parts or
        # the Orq router silently drops it, leaving the judge blind to the agent's
        # replies (RES-1308). Never hand-build this list.
        input_messages = messages_to_responses_input(messages)
        # A per-item breakpoint, not the top-level switch: the latter marks the end
        # of the whole input, so a caller that rebuilds its trailing item (the
        # judge) writes every turn and reads none. `volatile_tail` counts messages
        # and this list counts items, which are not 1:1 — one tool-calling assistant
        # message renders to several items.
        if caching_applies(self._client, self._model):
            input_messages = mark_responses_input(
                input_messages,
                volatile_items=responses_volatile_items(messages, volatile_tail=volatile_tail),
            )

        async with with_llm_span(
            model=self._model,
            operation='responses',
            temperature=resolved_temp,
            max_tokens=max_tok,
            purpose=llm_purpose,
        ) as span:
            # Responses API sends system context via `instructions`, not as a
            # message in `input`. Record what is actually sent so the span
            # matches the real request shape (mirrors _call_chat_completions
            # which records full_messages including the system entry).
            if span is not None:
                span.set_attribute('gen_ai.request.instructions', self.system_prompt[:2000])
            # Record from `messages`, NOT from `input_messages`: the wire payload
            # renders an assistant turn as `[{'type': 'output_text', ...}]`, which
            # `record_llm_input` would `str()` into a Python repr on the span.
            # Mirrors runner/simulation.py's target_call span. See CLAUDE.md's
            # `content_to_text` row. `record_input=False` below keeps
            # `execute_response` from overwriting it with the wire shape.
            record_llm_input(span, [{'role': m.role, 'content': span_message_text(m.content)} for m in messages])

            # Config body layered last: the caller's keys win per key, the call
            # site's thread grouping survives the ones they did not set.
            extra_body = thread_body_param() if client_routes_through_orq(self._client) else {}
            if self.config.extra_body:
                extra_body = {**extra_body, **self.config.extra_body}

            async def _do_call() -> LLMResult:
                response, usage = await execute_response(
                    client=self._client,
                    model=self._model,
                    messages=input_messages,
                    span=span,
                    timeout_s=timeout_s,
                    temperature=resolved_temp,
                    max_output_tokens=max_tok,
                    reasoning_effort=reasoning_effort,
                    instructions=self.system_prompt,
                    tools=[_responses_tool_schema(tool) for tool in tools] if tools else None,
                    record_input=False,
                    extra_body=extra_body or None,
                    extra_kwargs=self.config.extra_kwargs or None,
                )
                self._accumulate(usage)

                stop_reason = responses_stop_reason(response)
                if stop_reason == 'length':
                    raise RuntimeError(
                        f'{self.name}._call_responses: response truncated at max_output_tokens='
                        f'{max_tok}; {self._max_tokens_advice(max_tokens)}.'
                    )
                refusal = first_responses_refusal(response)
                output_items = AgentResponse.from_openresponses(response).output

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
                        'If reason=max_output_tokens, %s.',
                        self.name,
                        self.config.model,
                        reason,
                        self._max_tokens_advice(max_tokens),
                    )

                text = ''.join(getattr(i, 'text', '') for i in text_items)
                result = LLMResult(content=text, refusal=refusal)
                if tool_call_items:
                    result.tool_calls = [
                        StrategyToolCall(
                            id=item.call_id,
                            function=FunctionCall(name=item.name, arguments=item.arguments),
                        )
                        for item in tool_call_items
                    ]
                return result

            return await with_retry(
                _do_call,
                label=f'{self.name}._call_responses',
                max_attempts=self.config.retry_count + 1,
            )


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
