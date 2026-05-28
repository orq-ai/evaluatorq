"""Backend abstract base class for dynamic red teaming agent targets.

Defines the ``Backend`` ABC plus the ``BareTargetBackend`` adapter. The
``AgentTarget`` ABC that backends construct lives in ``evaluatorq.contracts``
(relocated in RES-808 PR2). The ORQ implementation lives in ``backends.orq``;
other backends (HTTP, LangChain, custom callables) subclass ``Backend``
independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import AgentResponse

if TYPE_CHECKING:
    from evaluatorq.contracts import AgentContext, AgentTarget


def _coerce_to_agent_response(raw: Any) -> AgentResponse:
    """Wrap a plain str return into AgentResponse for backward-compat with legacy targets.

    Any target that still returns ``str`` from ``send_prompt`` will be transparently
    wrapped here at the orchestrator call site.
    """
    from evaluatorq.redteam.contracts import OutputMessage, TextOutputItem
    if isinstance(raw, AgentResponse):
        return raw
    text_item: OutputMessage = TextOutputItem(text=str(raw) if raw is not None else '', annotations=[])
    return AgentResponse(output=[text_item])


def validate_agent_target(obj: object) -> None:
    """Raise ``TypeError`` if ``obj`` implements only the removed ``clone()`` API.

    The check fires only when the object has ``clone()`` but neither
    ``send_prompt`` nor ``new()`` — i.e. a clone-only object that cannot be
    used as an :class:`AgentTarget` at all. Objects that implement the full
    protocol (``send_prompt`` + ``new``) are accepted regardless of whether
    they also define ``clone``.
    """
    has_send = callable(getattr(obj, 'send_prompt', None))
    has_new = callable(getattr(obj, 'new', None))
    if not has_send and not has_new and callable(getattr(obj, 'clone', None)):
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
        return "target_error", f"{type(exc).__name__}: {exc}"

    async def resolve_context(self, agent_key: str) -> AgentContext:
        """Resolve agent context for a key by probing a target once, then caching.

        Distinct from ``AgentTarget.get_agent_context()`` (no args, self-describing):
        a backend resolves context for *any* key, a target describes *itself*.
        """
        if agent_key in self._ctx_cache:
            return self._ctx_cache[agent_key]
        probe = self.create_target(agent_key)
        ctx = await probe.get_agent_context()
        self._ctx_cache[agent_key] = ctx
        return ctx


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
                f"BareTargetBackend: {type(self._target).__name__} created "
                f"{len(entity_ids)} memory entity id(s) but does not override cleanup_memory; "
                "they may persist. Implement cleanup_memory on the target to release them."
            )
        await self._target.cleanup_memory(ctx, entity_ids)

    def map_error(self, exc: Exception) -> tuple[str, str]:
        result = self._target.map_error(exc)
        return result if result is not None else super().map_error(exc)
