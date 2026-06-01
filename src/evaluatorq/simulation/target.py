"""Stateless OrqResponsesTarget — implements the AgentTarget.respond interface."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import AgentContext, AgentResponse, AgentTarget, LLMCallConfig, Message
from evaluatorq.simulation._client import build_simulation_client, extract_responses_output
from evaluatorq.simulation.tracing import (
    record_openresponses_request,
    record_openresponses_response,
    with_llm_span,
)
from evaluatorq.common.fields import get_field as _get_field
from evaluatorq.simulation.utils.retry import with_retry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openai import AsyncOpenAI


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
        retry_attempts: int | None = None,
        retry_statuses: Iterable[int] | None = None,
    ) -> None:
        super().__init__(memory_entity_id=memory_entity_id)
        self.config = config
        self.instructions = instructions
        self.tools = tools
        self.retry_attempts = retry_attempts
        self.retry_statuses = set(retry_statuses) if retry_statuses is not None else None
        if client is not None:
            self._client = client
            self._client_owned = False
        else:
            self._client, self._client_owned = build_simulation_client(config.client)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Stateless: send the full message list, return the response."""
        return await self._call_responses_api(
            responses_input=self._messages_to_input(messages),
        )

    def new(self) -> OrqResponsesTarget:
        """Return a fresh instance with identical config but no shared state.

        Externally-injected clients (``_client_owned=False``) are propagated to
        the new instance so callers sharing a single HTTP connection continue to
        do so. Self-owned clients are not propagated — the new instance builds
        its own from env vars, keeping connection lifetimes independent.

        Per the ``AgentTarget`` contract each ``new()`` produces an independent
        memory scope; we mint a fresh ``memory_entity_id`` if one was set.
        """
        fresh_memory_id = str(uuid.uuid4()) if self.memory_entity_id is not None else None
        return OrqResponsesTarget(
            self.config,
            instructions=self.instructions,
            tools=self.tools,
            memory_entity_id=fresh_memory_id,
            client=self._client if not self._client_owned else None,
            retry_attempts=self.retry_attempts,
            retry_statuses=self.retry_statuses,
        )

    async def get_agent_context(self) -> AgentContext:
        """Describe this target — the configured model is the agent key."""
        return AgentContext(key=self.config.model, model=self.config.model)

    async def close(self) -> None:
        """Close the underlying HTTP client if this instance owns it.

        Externally-injected clients (``_client_owned=False``) are left
        untouched — the caller owns their lifecycle. Safe to call repeatedly.
        """
        if self._client_owned:
            await self._client.close()
            self._client_owned = False

    async def __aenter__(self) -> OrqResponsesTarget:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def _call_responses_api(
        self,
        *,
        responses_input: str | list[dict[str, Any]],
    ) -> AgentResponse:
        """Pure call into ``client.responses.create``; no instance mutation.

        Applies retry (rate-limit / server errors) via :func:`with_retry` and
        converts :class:`asyncio.TimeoutError` into a descriptive RuntimeError.
        """
        timeout_s = self.config.timeout_ms / 1000.0 if self.config.timeout_ms else None

        async def _do_call() -> AgentResponse:
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

            output_items, usage = extract_responses_output(response)
            if not output_items:
                raise RuntimeError(
                    f"OrqResponsesTarget: response contained no extractable "
                    f"output items (model={self.config.model}). This likely indicates "
                    f"an API error or unexpected response format."
                )
            finish_reason = _get_field(response, "status")
            response_model = _get_field(response, "model")

            # extract_responses_output returns calls=0; this was one API call.
            # Bump it so usage aggregation stays consistent with the other targets.
            if usage is not None:
                usage = usage.model_copy(update={"calls": 1})
            return AgentResponse(
                output=output_items,
                usage=usage,
                model=response_model if isinstance(response_model, str) else self.config.model,
                finish_reason=finish_reason if isinstance(finish_reason, str) else None,
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
    def _messages_to_input(messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for m in messages:
            # Assistant messages with tool_calls require content: null per
            # OpenAI's spec; collapsing None to "" produces an invalid payload.
            # For other roles, the API expects a string, so coerce None -> "".
            if m.role == "assistant":
                content: Any = m.content
            else:
                content = m.content or ""
            d: dict[str, Any] = {"role": m.role, "content": content}
            if m.tool_calls is not None:
                d["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                d["name"] = m.name
            result.append(d)
        return result




__all__ = ["OrqResponsesTarget"]
