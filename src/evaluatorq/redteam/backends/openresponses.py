"""OpenResponses backend wrapping the shared simulation Responses target."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.contracts import AgentContext, LLMCallConfig
from evaluatorq.redteam.backends._errors import (
    extract_provider_error_code,
    extract_status_code,
)
from evaluatorq.redteam.backends.base import Backend
from evaluatorq.simulation.target import OrqResponsesTarget

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenResponsesBackend(Backend):
    """Backend for OpenResponses targets backed by the simulation Responses target.

    Targets are server-side stateful via ``previous_response_id`` threading,
    but the server owns the memory lifecycle — ``cleanup_memory`` is a no-op
    because we cannot delete prior responses by id.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        instructions: str | None = None,
        timeout_ms: int | None = None,
        retry_attempts: int | None = None,
        retry_statuses: list[int] | None = None,
    ) -> None:
        super().__init__(name="openresponses")
        self._client = client
        self._instructions = instructions
        self._timeout_ms = timeout_ms
        self._retry_attempts = retry_attempts
        self._retry_statuses = retry_statuses

    def create_target(self, agent_key: str) -> OrqResponsesTarget:
        # OrqResponsesTarget picks up the client from the explicit ``client=``
        # kwarg below; ``config.client`` is left ``None`` so the precedence is
        # unambiguous.
        config = LLMCallConfig(
            model=agent_key,
            api="responses",
            timeout_ms=self._timeout_ms or 240_000,
        )
        return OrqResponsesTarget(
            config,
            instructions=self._instructions,
            client=self._client,
            retry_attempts=self._retry_attempts,
            retry_statuses=self._retry_statuses,
        )

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        logger.debug("OpenResponses backend has no client-side memory store; cleanup is a no-op")

    def map_error(self, exc: Exception) -> tuple[str, str]:
        status_code = extract_status_code(exc)
        if status_code is not None:
            return f"openresponses.http.{status_code}", f"{type(exc).__name__}: {exc}"
        provider_code = extract_provider_error_code(exc)
        if provider_code:
            return f"openresponses.code.{provider_code}", f"{type(exc).__name__}: {exc}"
        name = type(exc).__name__.lower()
        if "ratelimit" in name:
            return "openresponses.rate_limit", f"{type(exc).__name__}: {exc}"
        if "timeout" in name:
            return "openresponses.timeout", f"{type(exc).__name__}: {exc}"
        if "authentication" in name:
            return "openresponses.auth", f"{type(exc).__name__}: {exc}"
        return "openresponses.unknown", f"{type(exc).__name__}: {exc}"

    async def resolve_context(self, agent_key: str) -> AgentContext:
        if agent_key in self._ctx_cache:
            return self._ctx_cache[agent_key]
        ctx = AgentContext(
            key=agent_key,
            display_name=agent_key,
            description="OpenResponses agent target",
            system_prompt=self._instructions,
            model=agent_key,
        )
        self._ctx_cache[agent_key] = ctx
        return ctx
