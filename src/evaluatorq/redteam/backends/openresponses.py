"""OpenResponses backend wrapping the shared simulation Responses target."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.contracts import AgentContext, LLMCallConfig
from evaluatorq.openresponses.target import OrqResponsesTarget
from evaluatorq.redteam.backends._errors import (
    extract_provider_error_code,
    extract_status_code,
)
from evaluatorq.redteam.backends.base import Backend

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenResponsesBackend(Backend):
    """Backend for OpenResponses targets backed by the simulation Responses target.

    Targets are stateless per call: ``OrqResponsesTarget`` sends the full
    transcript on every ``respond()`` and never populates
    ``previous_response_id`` (see ``openresponses/types.py``'s
    ``ResponseResourceDict.previous_response_id``, which models the field but
    nothing sets it). ``cleanup_memory`` is a no-op because the memory scope
    referenced by ``memory_entity_id`` is owned and expired server-side; we
    cannot delete it by id from here.

    This backend exclusively serves hosted ORQ ``agent/<key>`` targets, whose
    model id only resolves on the Orq router. ``require_orq`` therefore defaults
    to True: when no client is injected and the target builds its own from the
    env, an ``OPENAI_API_KEY`` left in the environment must never capture it.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        instructions: str | None = None,
        timeout_ms: int | None = None,
        retry_attempts: int = 1,
        retry_statuses: list[int] | None = None,
        reasoning_effort: str | None = None,
        require_orq: bool = True,
    ) -> None:
        super().__init__(name='openresponses')
        self._client = client
        self._instructions = instructions
        self._timeout_ms = timeout_ms
        self._retry_attempts = retry_attempts
        self._retry_statuses = retry_statuses
        self._reasoning_effort = reasoning_effort
        self._require_orq = require_orq

    def create_target(self, agent_key: str) -> OrqResponsesTarget:
        # OrqResponsesTarget picks up the client from the explicit ``client=``
        # kwarg below; ``config.client`` is left ``None`` so the precedence is
        # unambiguous.
        config = LLMCallConfig(
            model=agent_key,
            api='responses',
            timeout_ms=self._timeout_ms or 240_000,  # 240s matches Orq router default for long-tail tool calls
            reasoning_effort=self._reasoning_effort,
        )
        return OrqResponsesTarget(
            config,
            instructions=self._instructions,
            client=self._client,
            retry_attempts=self._retry_attempts,
            retry_statuses=self._retry_statuses,
            require_orq=self._require_orq,
        )

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        logger.debug('OpenResponses backend has no client-side memory store; cleanup is a no-op')

    def map_error(self, exc: Exception) -> tuple[str, str]:
        status_code = extract_status_code(exc)
        if status_code is not None:
            return f'openresponses.http.{status_code}', f'{type(exc).__name__}: {exc}'
        provider_code = extract_provider_error_code(exc)
        if provider_code:
            return f'openresponses.code.{provider_code}', f'{type(exc).__name__}: {exc}'
        name = type(exc).__name__.lower()
        if 'ratelimit' in name:
            return 'openresponses.rate_limit', f'{type(exc).__name__}: {exc}'
        if 'timeout' in name:
            return 'openresponses.timeout', f'{type(exc).__name__}: {exc}'
        if 'authentication' in name:
            return 'openresponses.auth', f'{type(exc).__name__}: {exc}'
        logger.opt(exception=exc).error(
            'OpenResponsesBackend.map_error: unclassified exception mapped to openresponses.unknown'
        )
        return 'openresponses.unknown', f'{type(exc).__name__}: {exc}'

    async def resolve_context(self, agent_key: str) -> AgentContext:
        if agent_key in self._ctx_cache:
            return self._ctx_cache[agent_key]
        ctx = AgentContext(
            key=agent_key,
            display_name=agent_key,
            description='OpenResponses agent target',
            system_prompt=self._instructions or '',
            model=agent_key,
        )
        self._ctx_cache[agent_key] = ctx
        return ctx
