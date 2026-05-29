"""OpenResponses backend for dynamic red teaming (RES-540).

Target adapter that drives the agent / deployment through the OpenResponses
``/responses`` API. The request goes over the wire in the format spelled
out in the RES-540 ticket::

    {"model": "agent-id",
     "input": [{"role": "user", "content": "adversarial prompt here"}]}

For multi-turn attacks, the conversation grows turn-by-turn using
``previous_response_id`` (server-side threading, preferred) with a
client-side ``input``-array fallback when the server omits the response
id. Both paths use the OpenResponses input shape — never OpenAI
chat-completions format.

Trace spans are emitted in OpenResponses shape via
:func:`evaluatorq.redteam.openresponses_adapter.record_openresponses_request`
and :func:`record_openresponses_response`, so observability shows the
exact payload sent and received instead of a translated OpenAI view.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import AgentResponse, OutputMessage
from evaluatorq.redteam.backends._errors import extract_provider_error_code, extract_status_code
from evaluatorq.redteam.contracts import (
    DEFAULT_TARGET_MAX_TOKENS,
    DEFAULT_TARGET_TIMEOUT_MS,
    AgentContext,
    TargetKind,
    TokenUsage,
)
from evaluatorq.redteam.openresponses_adapter import (
    append_assistant_turn,
    append_user_followup,
    build_openresponses_request,
    record_openresponses_request,
    record_openresponses_response,
    user_input_item,
)
from evaluatorq.redteam.tracing import with_llm_span
from evaluatorq.simulation._client import extract_responses_output

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def create_openresponses_client() -> AsyncOpenAI:
    """Build an ``AsyncOpenAI`` client routed at the ORQ ``/responses`` endpoint.

    Defined as a module-level function so tests can monkeypatch the factory
    without touching the registry. Mirrors the routing rules used by
    ``simulation/_client.py``.
    """
    from evaluatorq.simulation._client import build_simulation_client

    client, _ = build_simulation_client()
    return client


# Backwards-compatible alias for callers that imported the private name during
# RES-540 development.
_create_openresponses_client = create_openresponses_client


def _map_openresponses_error(exc: Exception) -> tuple[str, str]:
    """Single source of truth for OpenResponses error taxonomy.

    Both :meth:`OpenResponsesAgentTarget.map_error` and
    :class:`OpenResponsesErrorMapper` delegate here so changes only need to
    be made in one place.
    """
    status_code = extract_status_code(exc)
    if status_code is not None:
        return f"openresponses.http.{status_code}", f"{type(exc).__name__}: {exc}"
    provider_code = extract_provider_error_code(exc)
    if provider_code:
        return f"openresponses.code.{provider_code}", f"{type(exc).__name__}: {exc}"
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return "openresponses.rate_limit", f"{type(exc).__name__}: {exc}"
    if "timeout" in name or isinstance(exc, asyncio.TimeoutError):
        return "openresponses.timeout", f"{type(exc).__name__}: {exc}"
    if "authentication" in name:
        return "openresponses.auth", f"{type(exc).__name__}: {exc}"
    return "openresponses.unknown", f"{type(exc).__name__}: {exc}"


class OpenResponsesAgentTarget:
    """Target adapter that calls an agent/deployment via the OpenResponses API.

    Implements the :class:`evaluatorq.redteam.backends.base.AgentTarget`
    protocol so it slots into the existing red teaming orchestrator without
    further changes. The orchestrator still calls ``send_prompt(str)``;
    internally this target packages the prompt into the OpenResponses
    request shape and threads multi-turn state by ``previous_response_id``.

    Args:
        agent_id: The ``model`` value sent to the OpenResponses API
            (typically the platform agent id or deployment key).
        client: Optional pre-configured ``AsyncOpenAI`` client. When ``None``,
            one is built via :func:`_create_openresponses_client`.
        instructions: Optional system instructions sent on every request as
            the OpenResponses top-level ``instructions`` field.
        tools: Optional list of tool definitions in OpenResponses tool shape.
        max_tokens: Map to ``max_output_tokens`` on the request.
        timeout_ms: Per-call timeout.
        use_server_threading: When ``True`` (default), threads multi-turn
            state via ``previous_response_id``. When ``False`` (or when the
            server omits ``id``), threads client-side by sending the full
            ``input`` array on each call.

    Attributes:
        memory_entity_id: Always ``None`` — the OpenResponses backend does
            not allocate platform-side memory entities for red teaming
            sessions; conversation state lives in ``previous_response_id``
            or the client-side ``input`` array.
    """

    memory_entity_id: str | None = None
    target_kind: TargetKind = TargetKind.OPENAI  # routed through the OpenAI-compatible /responses path

    def __init__(
        self,
        agent_id: str,
        *,
        client: AsyncOpenAI | None = None,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
        use_server_threading: bool = True,
    ):
        self.agent_id = agent_id
        self.client = client or create_openresponses_client()
        self.instructions = instructions
        self.tools = list(tools) if tools else None
        self.max_tokens = max_tokens or DEFAULT_TARGET_MAX_TOKENS
        self.timeout_ms = timeout_ms or DEFAULT_TARGET_TIMEOUT_MS
        self.use_server_threading = use_server_threading

        # Multi-turn state — only one of these is consulted per request.
        # ``_previous_response_id`` carries the server-side thread; if the
        # server omits ``id`` we fall back to the client-side ``input`` log.
        self._previous_response_id: str | None = None
        self._client_side_input: list[dict[str, Any]] = []
        self._threading_disabled: bool = False

    @property
    def name(self) -> str:
        return self.agent_id

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Send ``prompt`` to the OpenResponses ``/responses`` endpoint.

        Builds the payload in the ticket-spec shape, emits OpenResponses-
        shaped trace attributes, and updates conversation state so the
        next call continues the same thread.
        """
        previous_id = self._previous_response_id if self.use_server_threading else None
        use_client_thread = (
            not self.use_server_threading or self._threading_disabled or previous_id is None
        )

        if use_client_thread:
            input_array = list(self._client_side_input)
            input_array.append(user_input_item(prompt))
            payload_prompt = None
            conversation = input_array
        else:
            # When threading server-side, send only the new user turn — the
            # platform reconstructs the conversation from previous_response_id.
            payload_prompt = prompt
            conversation = None

        extra: dict[str, Any] = {"max_output_tokens": self.max_tokens}
        if self.tools:
            extra["tools"] = self.tools
        if previous_id and not use_client_thread:
            extra["previous_response_id"] = previous_id

        payload = build_openresponses_request(
            model=self.agent_id,
            prompt=payload_prompt,
            conversation=conversation,
            instructions=self.instructions,
            extra=extra,
        )

        async with with_llm_span(
            model=self.agent_id,
            operation="responses",
            attributes={"orq.redteam.llm_purpose": "target"},
        ) as span:
            record_openresponses_request(span, payload)

            response = await asyncio.wait_for(
                self.client.responses.create(**payload),
                timeout=self.timeout_ms / 1000.0,
            )

            output_items, usage = extract_responses_output(response)
            agent_usage = self._to_redteam_usage(usage)

            response_id = getattr(response, "id", None)
            finish_reason = getattr(response, "status", None)
            agent_response = AgentResponse(
                output=list(output_items),
                usage=agent_usage,
                model=getattr(response, "model", None) or self.agent_id,
                response_id=response_id if isinstance(response_id, str) else None,
                finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            )

            record_openresponses_response(span, agent_response)

        # Advance conversation state for the next turn.
        #
        # We mirror every successful turn into ``_client_side_input`` even when
        # server-side threading is active. If the server later stops returning
        # a response_id mid-conversation, the fallback path then has the full
        # prior history rather than just the failed turn — a partial history
        # would silently restart the conversation with degraded context.
        # Cost: ~2x memory for multi-turn sessions; acceptable given red team
        # sessions are bounded by max_turns (default 5).
        if self.use_server_threading and isinstance(response_id, str) and response_id:
            self._previous_response_id = response_id
            append_user_followup(self._client_side_input, prompt)
            append_assistant_turn(self._client_side_input, agent_response)
        else:
            if self.use_server_threading and not self._threading_disabled:
                logger.warning(
                    "OpenResponsesAgentTarget: response missing 'id'; falling back to "
                    "client-side input threading for agent_id={}",
                    self.agent_id,
                )
                self._threading_disabled = True
            # Replay the user turn we sent and append the assistant reply so the
            # next call can resend the full input array.
            append_user_followup(self._client_side_input, prompt)
            append_assistant_turn(self._client_side_input, agent_response)

        return agent_response

    def new(self) -> OpenResponsesAgentTarget:
        """Return a fresh target instance with no carried-over conversation state."""
        return OpenResponsesAgentTarget(
            agent_id=self.agent_id,
            client=self.client,
            instructions=self.instructions,
            tools=self.tools,
            max_tokens=self.max_tokens,
            timeout_ms=self.timeout_ms,
            use_server_threading=self.use_server_threading,
        )

    def create_target(self, agent_key: str) -> OpenResponsesAgentTarget:
        return OpenResponsesAgentTarget(
            agent_id=agent_key,
            client=self.client,
            instructions=self.instructions,
            tools=self.tools,
            max_tokens=self.max_tokens,
            timeout_ms=self.timeout_ms,
            use_server_threading=self.use_server_threading,
        )

    async def get_agent_context(self) -> AgentContext:
        return AgentContext(
            key=self.agent_id,
            display_name=self.agent_id,
            description="OpenResponses agent target",
            system_prompt=self.instructions,
            model=self.agent_id,
        )

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Normalize OpenResponses transport exceptions for the runner error taxonomy."""
        return _map_openresponses_error(exc)

    @staticmethod
    def _to_redteam_usage(usage: Any) -> TokenUsage | None:
        """Convert the simulation ``TokenUsage`` shape to the redteam variant.

        ``extract_responses_output`` returns ``evaluatorq.simulation.types.TokenUsage``
        which has the same field names but is a distinct class — the
        orchestrator's accumulators expect the redteam :class:`TokenUsage`.

        When the upstream ``total_tokens`` field is absent or zero, falls back
        to ``prompt + completion`` so token-cost dashboards never display a
        spurious zero for a billed call.
        """
        if usage is None:
            return None
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        raw_total = int(getattr(usage, "total_tokens", 0) or 0)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=raw_total or (prompt_tokens + completion_tokens),
            calls=1,
        )

    @property
    def output_messages(self) -> list[OutputMessage]:
        """Always empty — kept on the class for protocol parity only.

        OpenResponses conversation state lives in ``previous_response_id``
        (server-side threading) or the ``_client_side_input`` array
        (fallback path), never in a cached output list. Callers that need
        the most recent response should keep the :class:`AgentResponse`
        returned by :meth:`send_prompt` directly.
        """
        return []


class OpenResponsesContextProvider:
    """Minimal context provider for OpenResponses targets.

    The OpenResponses backend treats the agent id as opaque — tools / memory /
    knowledge metadata are owned by the platform-side agent definition and are
    not introspected client-side. Returns a basic :class:`AgentContext` so
    strategy planning and capability classification have something to anchor
    on.
    """

    def __init__(self, instructions: str | None = None):
        self._instructions = instructions

    async def get_agent_context(self, agent_key: str) -> AgentContext:
        logger.debug("Using OpenResponses target context for agent_id={}", agent_key)
        return AgentContext(
            key=agent_key,
            display_name=agent_key,
            description="OpenResponses agent target",
            system_prompt=self._instructions,
            tools=[],
            memory_stores=[],
            knowledge_bases=[],
            model=agent_key,
        )


class OpenResponsesTargetFactory:
    """Factory that produces :class:`OpenResponsesAgentTarget` instances per job."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
        use_server_threading: bool = True,
    ):
        self._client = client
        self._instructions = instructions
        self._tools = tools
        self._max_tokens = max_tokens
        self._timeout_ms = timeout_ms
        self._use_server_threading = use_server_threading

    def create_target(self, agent_key: str) -> OpenResponsesAgentTarget:
        return OpenResponsesAgentTarget(
            agent_id=agent_key,
            client=self._client,
            instructions=self._instructions,
            tools=self._tools,
            max_tokens=self._max_tokens,
            timeout_ms=self._timeout_ms,
            use_server_threading=self._use_server_threading,
        )


class OpenResponsesErrorMapper:
    """Standalone error mapper so the backend bundle has a fresh mapper per call.

    Delegates to :func:`_map_openresponses_error` — the single source of
    truth for the OpenResponses error taxonomy.
    """

    def map_error(self, exc: Exception) -> tuple[str, str]:
        return _map_openresponses_error(exc)


class OpenResponsesBackend:
    """``Backend``-shaped wrapper for the OpenResponses target components.

    Satisfies the ``Backend`` ABC interface so the registry can return a
    uniform object regardless of which backend is selected. Delegates all
    behaviour to the composable factory / context-provider / error-mapper
    components so those can be tested independently.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        instructions: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        self.name = "openresponses"
        self.target_factory = OpenResponsesTargetFactory(
            client,
            instructions=instructions,
            timeout_ms=timeout_ms,
        )
        self.context_provider = OpenResponsesContextProvider(instructions=instructions)
        self.error_mapper = OpenResponsesErrorMapper()
        self._ctx_cache: dict[str, AgentContext] = {}

    def create_target(self, agent_key: str) -> OpenResponsesAgentTarget:
        return self.target_factory.create_target(agent_key)

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        pass  # OpenResponses backend has no server-side memory to clean up

    def map_error(self, exc: Exception) -> tuple[str, str]:
        return self.error_mapper.map_error(exc)

    async def resolve_context(self, agent_key: str) -> AgentContext:
        if agent_key in self._ctx_cache:
            return self._ctx_cache[agent_key]
        ctx = await self.context_provider.get_agent_context(agent_key)
        self._ctx_cache[agent_key] = ctx
        return ctx


__all__ = [
    "OpenResponsesAgentTarget",
    "OpenResponsesBackend",
    "OpenResponsesContextProvider",
    "OpenResponsesErrorMapper",
    "OpenResponsesTargetFactory",
    "create_openresponses_client",
]
