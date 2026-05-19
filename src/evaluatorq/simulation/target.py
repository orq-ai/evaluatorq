"""Simulation target implementations backed by the Orq Responses API."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import AgentResponse, LLMCallConfig
from evaluatorq.simulation._client import (
    _get_field,
    build_simulation_client,
    extract_responses_output,
)
from evaluatorq.simulation.tracing import (
    record_openresponses_request,
    record_openresponses_response,
    with_llm_span,
)
from evaluatorq.simulation.types import ChatMessage, TokenUsage
from evaluatorq.simulation.utils.retry import with_retry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openai import AsyncOpenAI


@dataclass(frozen=True)
class _ResponsesCallResult:
    """Internal result of a single Responses API call.

    Holds everything a caller might use to update state — but the call itself
    performs no mutation. Callers decide what (if anything) to persist.
    """

    response: AgentResponse
    response_id: str | None
    usage: TokenUsage | None


class OrqResponsesTarget:
    """Wraps the Orq Responses v3 API as a simulation target.

    Implements two interfaces with different state semantics:

    * ``__call__(messages)`` — sim callable shape. **Stateless w.r.t. self**:
      reads no state, writes no state. The sim runner owns conversation
      history (passed via ``messages``) and tracks usage independently.

    * ``send_prompt(prompt)`` — redteam ``AgentTarget`` protocol. **Stateful**:
      threads via ``self._previous_response_id`` and accumulates token usage on
      ``self._accumulated_usage``. Multi-turn server-side state continuity.

    Concurrency contract: a single instance is single-caller for the stateful
    path. Two concurrent ``send_prompt`` calls on the same instance race on
    ``_previous_response_id`` and ``_accumulated_usage`` — wrap with an external
    ``asyncio.Lock`` if you need concurrent invocation. The stateless ``__call__``
    path is safe to invoke concurrently because it mutates nothing on self.
    """

    memory_entity_id: str | None
    threading_disabled: bool

    def __init__(
        self,
        config: LLMCallConfig,
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        memory_entity_id: str | None = None,
        client: AsyncOpenAI | None = None,
        retry_attempts: int | None = None,
        retry_statuses: Iterable[int] | None = None,
    ) -> None:
        self.config = config
        self.instructions = instructions
        self.tools = tools
        self.memory_entity_id = memory_entity_id
        self.retry_attempts = retry_attempts
        self.retry_statuses = set(retry_statuses) if retry_statuses is not None else None
        self.threading_disabled = False
        self._previous_response_id: str | None = None
        self._accumulated_usage = TokenUsage()
        if client is not None:
            self._client = client
            self._client_owned = False
        else:
            self._client, self._client_owned = build_simulation_client(config.client)

    async def __call__(self, messages: list[ChatMessage]) -> str:
        """Sim target_callback shape. Sends full message list each turn.

        Stateless w.r.t. self: never reads or writes ``_previous_response_id``
        or ``_accumulated_usage``. Safe to call concurrently with itself.
        """
        result = await self._invoke_stateless(
            responses_input=self._messages_to_input(messages),
        )
        return result.response.text

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Redteam AgentTarget protocol shape. Threads previous_response_id."""
        return await self._invoke_stateful(responses_input=prompt)

    def new(self) -> OrqResponsesTarget:
        """Return a fresh instance with identical config but cleared state.

        Externally-injected clients (``_client_owned=False``) are propagated to
        the new instance so callers sharing a single HTTP connection continue to
        do so. Self-owned clients are not propagated — the new instance builds
        its own from env vars, keeping connection lifetimes independent.

        Per AgentTarget protocol, each ``new()`` call must produce an
        independent memory scope; we mint a fresh ``memory_entity_id`` if one
        was set.
        """
        fresh_memory_id = (
            str(uuid.uuid4()) if self.memory_entity_id is not None else None
        )
        return OrqResponsesTarget(
            self.config,
            instructions=self.instructions,
            tools=self.tools,
            memory_entity_id=fresh_memory_id,
            client=self._client if not self._client_owned else None,
            retry_attempts=self.retry_attempts,
            retry_statuses=self.retry_statuses,
        )

    def get_usage(self) -> TokenUsage:
        """Return cumulative token usage across ``send_prompt`` calls.

        Only the stateful path (``send_prompt``) contributes; sim ``__call__``
        is stateless and does not update this counter.
        """
        return self._accumulated_usage.model_copy()

    async def close(self) -> None:
        """Close the underlying HTTP client if this instance owns it.

        Externally-injected clients (``_client_owned=False``) are left
        untouched — the caller owns their lifecycle. Safe to call multiple
        times; subsequent calls are no-ops.
        """
        if self._client_owned:
            await self._client.close()
            self._client_owned = False

    async def __aenter__(self) -> "OrqResponsesTarget":  # noqa: PYI034, UP037
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def _invoke_stateless(
        self,
        *,
        responses_input: str | list[dict[str, Any]],
    ) -> _ResponsesCallResult:
        """Sim path: pure call, no instance mutation, no threading."""
        return await self._call_responses_api(
            responses_input=responses_input,
            previous_response_id=None,
        )

    async def _invoke_stateful(
        self,
        *,
        responses_input: str | list[dict[str, Any]],
    ) -> AgentResponse:
        """Redteam path: threads via self._previous_response_id, accumulates usage.

        Atomic mutation: builds the new ``_previous_response_id`` and
        ``_accumulated_usage`` locally and writes them in a single block after
        the API call succeeds. A partial-mutation window remains only across
        the two assignments below, which run with no awaits between them.
        """
        result = await self._call_responses_api(
            responses_input=responses_input,
            previous_response_id=self._previous_response_id,
        )

        if result.usage is None:
            new_usage = self._accumulated_usage
        else:
            new_usage = TokenUsage(
                prompt_tokens=self._accumulated_usage.prompt_tokens + result.usage.prompt_tokens,
                completion_tokens=self._accumulated_usage.completion_tokens + result.usage.completion_tokens,
                total_tokens=self._accumulated_usage.total_tokens + result.usage.total_tokens,
            )
        threading_disabled = result.response_id is None
        if not threading_disabled:
            self._previous_response_id = result.response_id
        elif not self.threading_disabled:
            logger.error(
                "OrqResponsesTarget: response missing 'id'; server-side "
                "conversation threading disabled for this instance "
                "(model={}). Multi-turn continuity lost; subsequent send_prompt "
                "calls will behave as turn 1.",
                self.config.model,
            )
        self.threading_disabled = self.threading_disabled or threading_disabled
        self._accumulated_usage = new_usage

        return result.response

    async def _call_responses_api(
        self,
        *,
        responses_input: str | list[dict[str, Any]],
        previous_response_id: str | None,
    ) -> _ResponsesCallResult:
        """Pure call into ``client.responses.create``. No instance mutation.

        Applies retry (rate-limit / server errors) via :func:`with_retry` and
        converts :class:`asyncio.TimeoutError` into a descriptive RuntimeError.
        """
        timeout_s = self.config.timeout_ms / 1000.0 if self.config.timeout_ms else None

        async def _do_call() -> _ResponsesCallResult:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": responses_input,
            }
            if self.config.max_tokens:
                kwargs["max_output_tokens"] = self.config.max_tokens
            if self.tools:
                kwargs["tools"] = self.tools
            if self.instructions is not None:
                kwargs["instructions"] = self.instructions
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id

            async with with_llm_span(
                model=self.config.model,
                operation="responses",
                purpose="target",
                max_tokens=self.config.max_tokens,
            ) as span:
                record_openresponses_request(span, kwargs)
                coro = self._client.responses.create(**kwargs)
                response = await (
                    asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro
                )
                record_openresponses_response(span, response)

            response_id = _validate_response_id(response, self.config.model)

            output_items, usage = extract_responses_output(response)
            if not output_items:
                raise RuntimeError(
                    f"OrqResponsesTarget: response contained no extractable "
                    f"output items (model={self.config.model}). This likely indicates "
                    f"an API error or unexpected response format."
                )
            finish_reason = _get_field(response, "status")
            response_model = _get_field(response, "model")

            return _ResponsesCallResult(
                response=AgentResponse(
                    output=output_items,
                    usage=usage,
                    model=response_model if isinstance(response_model, str) else self.config.model,
                    response_id=response_id,
                    finish_reason=finish_reason if isinstance(finish_reason, str) else None,
                ),
                response_id=response_id,
                usage=usage,
            )

        try:
            retry_kwargs: dict[str, Any] = {}
            if self.retry_attempts is not None:
                retry_kwargs["max_attempts"] = self.retry_attempts
            if self.retry_statuses is not None:
                retry_kwargs["retry_statuses"] = self.retry_statuses
            return await with_retry(
                _do_call,
                label="OrqResponsesTarget._call_responses_api",
                **retry_kwargs,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"OrqResponsesTarget timed out after {timeout_s}s "
                f"(model={self.config.model})"
            ) from e

    @staticmethod
    def _messages_to_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]


def _validate_response_id(response: Any, model: str) -> str | None:
    """Return the response id if usable for threading, else None + warn."""
    response_id = _get_field(response, "id")
    if isinstance(response_id, str) and response_id:
        return response_id
    logger.warning(
        "OrqResponsesTarget: response missing 'id'; "
        "conversation threading disabled (model={})",
        model,
    )
    return None


__all__ = ["OrqResponsesTarget"]
