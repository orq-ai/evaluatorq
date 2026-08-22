"""Stateless OrqResponsesTarget — implements the AgentTarget.respond interface."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import BadRequestError
from typing_extensions import Self

from evaluatorq.common.llm_call import (
    is_responses_reasoning_rejection,
    remember_responses_reasoning_rejection,
    strip_known_rejected_responses_reasoning,
)
from evaluatorq.common.llm_client import client_routes_through_orq
from evaluatorq.common.responses import first_responses_refusal, responses_stop_reason
from evaluatorq.common.retry import with_retry, without_client_retries
from evaluatorq.common.thread_context import pipeline_metadata, thread_body_param
from evaluatorq.common.tracing import get_trace_context_headers
from evaluatorq.contracts import AgentContext, AgentResponse, AgentTarget, LLMCallConfig, Message, ToolInfo
from evaluatorq.openresponses.client import build_simulation_client
from evaluatorq.openresponses.input_items import messages_to_responses_input
from evaluatorq.openresponses.tracing import (
    record_openresponses_request,
    record_openresponses_response,
    with_llm_span,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openai import AsyncOpenAI


def _raw_responses_create(client: Any) -> Any | None:
    """Return the SDK raw Responses creator when HTTP headers are available.

    The parsed OpenAI response does not expose Orq's ``x-orq-trace-*`` headers.
    Restrict this to the SDK's own raw-response wrapper so injected mock or
    OpenAI-compatible clients keep using their existing ``responses.create``
    surface unchanged.
    """
    raw_client = getattr(client, 'with_raw_response', None)
    if raw_client is None or not type(raw_client).__module__.startswith('openai.'):
        return None
    return raw_client.responses.create


def _tool_info(tool: dict[str, Any]) -> ToolInfo:
    """Map one Responses tool dict to a ``ToolInfo`` for the attack planner.

    Accepts both the flat Responses shape (``{'type': 'function', 'name': ...}``)
    and the nested Chat Completions shape (``{'function': {'name': ...}}``);
    callers pass either through to the SDK. Unnamed/unknown tool shapes keep
    their ``type`` as the name rather than being dropped silently.
    """
    nested = tool.get('function')
    spec: dict[str, Any] = nested if isinstance(nested, dict) else tool
    return ToolInfo(
        name=str(spec.get('name') or tool.get('type') or 'unknown'),
        description=spec.get('description'),
        parameters=spec.get('parameters'),
        action_type=tool.get('type'),
    )


class OrqResponsesTarget(AgentTarget):
    """Wraps the Orq Responses v3 API as a stateless ``AgentTarget``.

    Stateless: each ``respond(messages)`` call sends the full message list and
    holds no per-instance conversation state. Conversation continuity is owned
    by the caller — the sim runner or the red-team orchestrator passes the full
    transcript every turn. ``respond`` is the sole response method; callers
    own the conversation transcript.

    Because nothing is mutated on ``self``, a single instance is safe to invoke
    concurrently.
    """

    def __init__(
        self,
        config: LLMCallConfig,
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        memory_entity_id: str | None = None,
        client: AsyncOpenAI | None = None,
        retry_attempts: int = 1,
        retry_statuses: Iterable[int] | None = None,
        require_orq: bool = False,
    ) -> None:
        super().__init__(memory_entity_id=memory_entity_id)
        self.config = config
        self.instructions = instructions
        self.tools = tools
        self.retry_attempts = retry_attempts
        self.retry_statuses = set(retry_statuses) if retry_statuses is not None else None
        # ORQ-agent targets set this so a self-built env client never falls back to
        # OPENAI_API_KEY — an ``agent/<key>`` model only resolves on the Orq router.
        self.require_orq = require_orq
        if client is not None:
            # This target owns retry via with_retry, including when callers
            # inject an AsyncOpenAI client that was built with SDK retries.
            self._client = without_client_retries(client)
            self._client_owned = False
        else:
            # max_retries=0: this target owns retry via with_retry in
            # _call_responses_api, so the SDK layer must not stack a second
            # backoff loop under it (up to 4 x 3 HTTP attempts otherwise).
            self._client, self._client_owned = build_simulation_client(
                config.client, require_orq=require_orq, max_retries=0
            )

    @property
    def memory_entity_id(self) -> str | None:
        return self._memory_entity_id

    @memory_entity_id.setter
    def memory_entity_id(self, value: str | None) -> None:
        # Assignment is how callers seed an explicit entity id (the sim layer's
        # --memory-entity path); mark it so ``new()`` preserves it across
        # clones. ``new()``'s re-mint writes ``_memory_entity_id`` directly and
        # stays unseeded. Mirrors ``ORQAgentTarget``'s seeded-vs-minted split.
        self._memory_entity_id = value
        self._memory_entity_seeded = value is not None

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Stateless: send the full message list, return the response."""
        return await self._call_responses_api(
            responses_input=messages_to_responses_input(messages),
        )

    def new(self) -> OrqResponsesTarget:
        """Return a fresh instance with identical config but no shared state.

        Externally-injected clients (``_client_owned=False``) are propagated to
        the new instance so callers sharing a single HTTP connection continue to
        do so. Self-owned clients are not propagated — the new instance builds
        its own from env vars, keeping connection lifetimes independent.

        An explicitly seeded ``memory_entity_id`` (constructor arg or later
        assignment) is preserved so clones keep pointing at the seeded entity;
        an unseeded one is re-minted per clone, keeping parallel jobs in
        independent memory scopes. Mirrors ``ORQAgentTarget.new()``.
        """
        clone = OrqResponsesTarget(
            self.config,
            instructions=self.instructions,
            tools=self.tools,
            memory_entity_id=self.memory_entity_id if self._memory_entity_seeded else None,
            client=self._client if not self._client_owned else None,
            retry_attempts=self.retry_attempts,
            retry_statuses=self.retry_statuses,
            require_orq=self.require_orq,
        )
        if not self._memory_entity_seeded and self.memory_entity_id is not None:
            # Re-mint bypassing the seeding setter so grandchild clones keep
            # re-minting instead of inheriting this one as if it were seeded.
            clone._memory_entity_id = str(uuid.uuid4())
        return clone

    async def get_agent_context(self) -> AgentContext:
        """Describe this target — the configured model is the agent key.

        ``instructions`` is carried through so attack planners see the persona
        the model actually runs with; dropping it makes them plan against a
        generic assistant. Consumers read ``instructions or system_prompt``, so
        filling this one alone is enough.
        """
        return AgentContext(
            key=self.config.model,
            model=self.config.model,
            instructions=self.instructions or '',
            tools=[_tool_info(t) for t in self.tools or []],
        )

    async def close(self) -> None:
        """Close the underlying HTTP client if this instance owns it.

        Externally-injected clients (``_client_owned=False``) are left
        untouched — the caller owns their lifecycle. Safe to call repeatedly.
        """
        if self._client_owned:
            await self._client.close()
            self._client_owned = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def _call_responses_api(
        self,
        *,
        responses_input: str | list[dict[str, Any]],
    ) -> AgentResponse:
        """Pure call into ``client.responses.create``; no instance mutation.

        Converts `asyncio.TimeoutError` into a descriptive RuntimeError.

        ``retry_attempts`` defaults to 1 — a single attempt, no retry — because
        ``call_target_with_retry`` owns the target retry budget on every surface
        that drives a target (see ``AgentTarget``). Raise it only when calling
        ``respond`` directly, outside that wrapper; under it, the two budgets
        multiply.
        """
        timeout_s = self.config.timeout_ms / 1000.0 if self.config.timeout_ms else None

        async def _do_call() -> AgentResponse:
            # Metadata is a native Responses field on every endpoint. Threads are
            # an Orq-router extension, so direct OpenAI-compatible endpoints must
            # not receive them in ``extra_body``.
            metadata = pipeline_metadata()
            routes_through_orq = client_routes_through_orq(self._client)
            body_extra = thread_body_param() if routes_through_orq else {}
            # Agents with memory tools reject the call outright without a memory
            # scope ("memory_entity_id_required"), so forward ours when set.
            if routes_through_orq and self.memory_entity_id:
                body_extra['memory'] = {'entity_id': self.memory_entity_id}

            # ``request_params`` folds temperature/max_output_tokens/reasoning
            # and ``extra_kwargs`` (top_p, store, truncation, tool_choice, ...)
            # into one dict, ``extra_kwargs`` winning last so a caller-supplied
            # value overrides these computed ones. ``extra_body`` is one of the
            # structural keys ``request_params`` guards — a caller cannot
            # replace the router's thread/memory body wholesale via
            # ``extra_kwargs={'extra_body': ...}``; ``check_reserved_keys``
            # raises ``ValueError`` locally at param-build time instead. The
            # router body assembled above (``body_extra``) is passed in as a
            # call-site param rather than merged in after the fact, so
            # ``_merge_extra_body`` stays the single place precedence is
            # decided: this config's ``extra_body`` — the public seam for a
            # caller who wants to add to it — is layered on top of
            # ``body_extra`` per key, so a caller-supplied key wins a clash
            # (e.g. scoping to a specific memory entity) while any router key
            # the config does not mention still survives.
            call_params: dict[str, Any] = {'input': responses_input}
            if self.tools:
                call_params['tools'] = self.tools
            if self.instructions is not None:
                call_params['instructions'] = self.instructions
            if metadata:
                call_params['metadata'] = metadata
            if body_extra:
                call_params['extra_body'] = body_extra
            kwargs: dict[str, Any] = {
                'model': self.config.model,
                **self.config.request_params(api='responses', **call_params),
            }
            # Drop the `reasoning` block up front if this model already 400'd on
            # it this process — same memo `common.llm_call.execute_response` uses,
            # so a rejection learned via the pipeline's own calls also short-circuits
            # target calls, and vice versa.
            strip_known_rejected_responses_reasoning(self.config.model, kwargs)

            async with with_llm_span(
                model=self.config.model,
                operation='responses',
                purpose='target',
                max_tokens=self.config.max_tokens,
            ) as span:
                # Propagate W3C trace context so the Orq router nests the
                # server-side agent trace under this target-call span (same
                # trace as the pipeline) instead of starting a loose root
                # trace. Captured inside the span so `traceparent` points at
                # it. Mirrors the user-simulator / judge / first-message calls.
                trace_headers = await get_trace_context_headers()
                if trace_headers:
                    kwargs['extra_headers'] = {**kwargs.get('extra_headers', {}), **trace_headers}
                record_openresponses_request(span, kwargs)
                raw_create = _raw_responses_create(self._client)

                async def _create() -> tuple[Any, Any | None]:
                    """Return ``(response, response_headers)`` for the current ``kwargs``."""
                    if raw_create is not None:
                        raw_coro = raw_create(**kwargs)
                        raw_response = await (asyncio.wait_for(raw_coro, timeout=timeout_s) if timeout_s else raw_coro)
                        return raw_response.parse(), raw_response.headers
                    coro = self._client.responses.create(**kwargs)
                    parsed = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)
                    return parsed, None

                try:
                    response, response_headers = await _create()
                except BadRequestError as exc:
                    # Same drop-and-retry-once contract as
                    # `common.llm_call.execute_response`: only a 400 naming
                    # `reasoning` in both the request and the error body is
                    # treated as a rejection; anything else propagates.
                    if not is_responses_reasoning_rejection(kwargs, exc):
                        raise
                    remember_responses_reasoning_rejection(self.config.model, kwargs)
                    logger.warning(
                        'OrqResponsesTarget: model {} rejected the reasoning block; dropping it and retrying once',
                        self.config.model,
                    )
                    kwargs.pop('reasoning', None)
                    response, response_headers = await _create()
                # Some SDK-compatible clients expose trace IDs on parsed response
                # telemetry. The Orq SDK raw-response path instead gets the
                # authoritative IDs from HTTP headers because its parser drops the
                # nonstandard telemetry field.
                telemetry = getattr(response, 'telemetry', None)
                trace_id = (
                    telemetry.get('trace_id') if isinstance(telemetry, dict) else getattr(telemetry, 'trace_id', None)
                )
                span_id = (
                    telemetry.get('span_id') if isinstance(telemetry, dict) else getattr(telemetry, 'span_id', None)
                )
                if response_headers is not None:
                    trace_id = response_headers.get('x-orq-trace-id') or trace_id
                    span_id = response_headers.get('x-orq-trace-span-id') or span_id
                record_openresponses_response(span, response)
                if span is not None and trace_id:
                    span.set_attribute('orq.trace_id', trace_id)

            stop_reason = responses_stop_reason(response)
            if stop_reason == 'length':
                raise RuntimeError(
                    f'OrqResponsesTarget: response truncated at max_output_tokens={self.config.max_tokens}; '
                    'raise the configured token budget and retry.'
                )
            refusal = first_responses_refusal(response)
            agent_response = AgentResponse.from_openresponses(response)
            if refusal is not None:
                agent_response = agent_response.model_copy(update={'refusal': refusal})
            if not agent_response.output:
                raise RuntimeError(
                    f'OrqResponsesTarget: response contained no extractable '
                    f'output items (model={self.config.model}). This likely indicates '
                    f'an API error or unexpected response format.'
                )
            updates: dict[str, Any] = {'model': agent_response.model or self.config.model}
            if trace_id:
                updates['trace_id'] = trace_id
            if span_id:
                updates['span_id'] = span_id
            if agent_response.usage is not None:
                updates['usage'] = agent_response.usage.with_calls(1)
            return agent_response.model_copy(update=updates)

        try:
            retry_kwargs: dict[str, Any] = {'max_attempts': self.retry_attempts}
            if self.retry_statuses is not None:
                retry_kwargs['retry_statuses'] = self.retry_statuses
            return await with_retry(
                _do_call,
                label='OrqResponsesTarget._call_responses_api',
                **retry_kwargs,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(f'OrqResponsesTarget timed out after {timeout_s}s (model={self.config.model})') from e


__all__ = ['OrqResponsesTarget']
