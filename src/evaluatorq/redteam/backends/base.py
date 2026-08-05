"""Backend abstract base class for dynamic red teaming agent targets.

Defines the ``Backend`` ABC plus the ``BareTargetBackend`` adapter. The
``AgentTarget`` ABC that backends construct lives in ``evaluatorq.contracts``
(relocated in RES-808 PR2). The ORQ implementation lives in ``backends.orq``;
other backends (HTTP, LangChain, custom callables) subclass ``Backend``
independently.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from evaluatorq.common.target_call import _coerce_to_agent_response as _coerce_to_agent_response
from evaluatorq.contracts import AgentContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from evaluatorq.contracts import AgentTarget


def validate_agent_target(obj: object) -> None:
    """Raise ``TypeError`` if ``obj`` implements only the removed ``clone()`` API.

    The check fires only when the object has ``clone()`` but neither
    ``respond`` nor ``new()`` — i.e. a clone-only object that cannot be
    used as an :class:`AgentTarget` at all. Objects that implement the full
    protocol (``respond`` + ``new``) are accepted regardless of whether
    they also define ``clone``.
    """
    has_respond = callable(getattr(obj, 'respond', None))
    has_new = callable(getattr(obj, 'new', None))
    if not has_respond and not has_new and callable(getattr(obj, 'clone', None)):
        raise TypeError(
            f"{type(obj).__name__} implements 'clone()' which was removed in evaluatorq 1.3. "
            "Rename it to 'new(self) -> AgentTarget' — signature is the same, no memory_entity_id param."
        )


class Backend(ABC):
    """Backend ABC. Owns target construction, memory cleanup, and error mapping.

    Subclasses must implement ``create_target`` and ``cleanup_memory``.
    ``map_error`` has a sensible default; override for provider-specific
    HTTP/status-code mapping.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._ctx_cache: dict[str, AgentContext] = {}

    @abstractmethod
    def create_target(self, agent_key: str) -> AgentTarget:
        """Create a new AgentTarget for the given agent key."""
        ...

    @abstractmethod
    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        """Delete memory entities created during a red teaming run."""
        ...

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Return normalized ``(error_code, error_message)``."""
        return 'target_error', f'{type(exc).__name__}: {exc}'

    async def resolve_context(self, agent_key: str) -> AgentContext:
        """Resolve agent context for a key by probing a target once, then caching.

        Distinct from ``AgentTarget.get_agent_context()`` (no args, self-describing):
        a backend resolves context for *any* key, a target describes *itself*.
        """
        if agent_key in self._ctx_cache:
            return self._ctx_cache[agent_key]
        probe = self.create_target(agent_key)
        try:
            ctx = await probe.get_agent_context()
        except ValidationError as e:
            # Context is enrichment (tool / memory-store / KB names for attack
            # prompts), not a prerequisite for attacking. Degrade to a minimal
            # context so provider metadata we cannot model never kills a run.
            #
            # Only ValidationError is caught, deliberately: an unreachable agent
            # (bad ORQ_API_KEY, wrong ORQ_BASE_URL, 404 on the key) must still
            # fail loudly — silently red-teaming nothing is worse than crashing.
            logger.warning(
                f'Could not model agent context for {agent_key}: {e} — '
                'continuing with minimal context; attack strategies will use limited context'
            )
            ctx = AgentContext(key=agent_key)
        self._ctx_cache[agent_key] = ctx
        return ctx


class HybridAgentBackend(Backend):
    """Composite backend for ORQ agent targets: ORQ SDK for context + cleanup,
    OrqResponses for execution.

    ``create_target`` prefixes the key with ``agent/`` so the OrqResponses
    Responses API invokes the hosted agent (server-side tools/memory/KB are then
    applied automatically). Context retrieval and memory cleanup stay on the ORQ
    SDK backend, which can actually introspect and delete.
    """

    def __init__(
        self,
        *,
        context_backend: Backend | None = None,
        context_backend_factory: Callable[[], Backend] | None = None,
        exec_backend: Backend,
    ) -> None:
        # Keyword-only: context and exec are the same type (Backend), so positional
        # args would let a caller silently swap them — routing cleanup to the exec
        # backend and execution to the SDK backend.
        #
        # The context backend may be supplied lazily via ``context_backend_factory``:
        # execution (``create_target``/``map_error``) never touches it, so a static
        # ``agent:`` run can avoid constructing the ORQ SDK backend entirely (and
        # thus avoid requiring ``orq-ai-sdk``). It is built on first context use.
        if (context_backend is None) == (context_backend_factory is None):
            raise ValueError('provide exactly one of context_backend / context_backend_factory')
        super().__init__(name='hybrid-agent')
        self._context_cached = context_backend
        self._context_factory = context_backend_factory
        self._exec = exec_backend

    @property
    def _context(self) -> Backend:
        """Resolve (and cache) the context backend, building it lazily if needed."""
        cached = self._context_cached
        if cached is not None:
            return cached
        factory = self._context_factory
        if factory is None:  # unreachable: __init__ guarantees exactly one source
            raise RuntimeError('HybridAgentBackend has no context backend configured')
        cached = factory()
        self._context_cached = cached
        return cached

    def create_target(self, agent_key: str) -> AgentTarget:
        """Delegate to exec backend with ``agent/`` prefix for OrqResponses routing."""
        # Strip any existing prefix first so an already-prefixed key does not become
        # ``agent/agent/<key>``.
        target = self._exec.create_target(f'agent/{agent_key.removeprefix("agent/")}')
        # Hosted agents with memory tools reject calls that carry no memory scope.
        # Mint one per target, matching ORQAgentTarget, so parallel jobs stay
        # isolated and cleanup_memory has an id to delete.
        if getattr(target, 'memory_entity_id', 'unset') is None:
            # Auto-mint, not a user seed: plain assignment would mark the id
            # seeded (clone-preserving), but a minted id must keep re-minting
            # per clone or parallel jobs share one memory scope.
            target.mint_memory_entity_id(f'red-team-{uuid.uuid4().hex[:12]}')
        return target

    async def resolve_context(self, agent_key: str) -> AgentContext:
        """Delegate context resolution to the ORQ SDK backend (has its own cache)."""
        return await self._context.resolve_context(agent_key)

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        """Delegate memory cleanup to the ORQ SDK backend."""
        await self._context.cleanup_memory(ctx, entity_ids)

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Delegate error taxonomy to the exec backend."""
        return self._exec.map_error(exc)


class BareTargetBackend(Backend):
    """Adapter wrapping a bare ``AgentTarget`` so it satisfies the ``Backend`` ABC.

    Used by the runner's bring-your-own-target path. Absorbs the duck-typed
    capability checks (``cleanup_memory``, ``map_error``) that used to scatter
    across ``runner.py``.
    """

    def __init__(self, target: AgentTarget) -> None:
        super().__init__(name=type(target).__name__)
        self._target = target

    def create_target(self, agent_key: str) -> AgentTarget:
        """Ignore ``agent_key`` — target was pre-configured at construction time."""
        return self._target.new()

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        # Local import: module-level would re-expose AgentTarget as backends.base.AgentTarget,
        # violating the clean-break invariant pinned by test_agent_target_not_re_exported_from_base.
        from evaluatorq.contracts import AgentTarget

        if entity_ids and type(self._target).cleanup_memory is AgentTarget.cleanup_memory:
            # Target created memory entities but inherits the no-op cleanup default;
            # adversarial data may persist. Surface loudly (matches ORQ path).
            logger.warning(
                f'BareTargetBackend: {type(self._target).__name__} created '
                f'{len(entity_ids)} memory entity id(s) but does not override cleanup_memory; '
                'they may persist. Implement cleanup_memory on the target to release them.'
            )
        await self._target.cleanup_memory(ctx, entity_ids)

    def map_error(self, exc: Exception) -> tuple[str, str]:
        result = self._target.map_error(exc)
        return result if result is not None else super().map_error(exc)
