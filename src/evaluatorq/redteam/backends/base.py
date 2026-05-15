"""Backend protocols for dynamic red teaming agent targets.

Defines the abstract interfaces that any agent backend must implement.
The ORQ implementation lives in ``backends.orq``; other backends (HTTP,
LangChain, custom callables) can implement these protocols independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from evaluatorq.redteam.contracts import AgentResponse

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import AgentContext, TokenUsage


class AgentTarget(Protocol):
    """Protocol for agent targets that can receive prompts.

    Targets that maintain persistent memory (server-side entities, LangGraph
    checkpointer threads, etc.) expose the isolation key as ``memory_entity_id``.
    Targets without persistent memory set it to ``None``. The red teaming
    pipeline reads this attribute after target creation to track entities for
    cleanup — it does not inject the value into the target.

    Backends must implement ``send_prompt`` returning :class:`AgentResponse`.
    ``AgentResponse`` carries both the ordered output items (text + tool calls)
    and the per-call LLM metadata (``usage``, ``model``, ``response_id``,
    ``finish_reason``) that previously lived on the now-removed ``SendResult``.
    """

    memory_entity_id: str | None

    async def send_prompt(self, prompt: str) -> 'AgentResponse':
        """Send a prompt and return the response (output items + token usage)."""
        ...

    def new(self) -> AgentTarget:
        """Return a fresh target instance with isolated state for a new attack.

        Each call must produce an independent instance — own ``memory_entity_id``
        for memory-backed targets, ``None`` otherwise.
        """
        ...


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


class SupportsClone(Protocol):
    """Optional hook for target cloning per parallel job.

    Note: The runner detects this via duck-typing (``getattr``/``callable``),
    not ``isinstance``.  Implementing this protocol is recommended for
    documentation and static analysis but not strictly required.

    Clones are expected to own their own ``memory_entity_id`` (fresh one when
    the target backs a persistent memory store; ``None`` otherwise).
    """

    def clone(self) -> AgentTarget:
        """Return a fresh target instance with isolated state."""
        ...


class SupportsTokenUsage(Protocol):
    """Optional hook for exposing token usage from last target call.

    Deprecated: ``AgentResponse.usage`` is the canonical channel for token usage.
    Implementing this protocol is no longer required.
    """

    def consume_last_token_usage(self) -> 'TokenUsage | None':
        """Return and clear usage from last call."""
        ...


class SupportsTargetMetadata(Protocol):
    """Optional hook for backend-specific target metadata."""

    def target_metadata(self) -> dict[str, object]:
        """Return provider/target metadata for diagnostics."""
        ...


class SupportsAgentContext(Protocol):
    """Optional: target provides its own agent context."""

    async def get_agent_context(self) -> AgentContext: ...


class SupportsTargetFactory(Protocol):
    """Optional: target can create per-job instances.

    The factory owns no memory-entity concern. Targets that manage persistent
    memory generate their own ``memory_entity_id`` when constructed; callers
    read it off the resulting target.
    """

    def create_target(self, agent_key: str) -> AgentTarget: ...


class SupportsMemoryCleanup(Protocol):
    """Optional: target can clean up memory entities."""

    async def cleanup_memory(self, agent_context: AgentContext, entity_ids: list[str]) -> None: ...


class SupportsErrorMapping(Protocol):
    """Optional: target provides custom error classification."""

    def map_error(self, exc: Exception) -> tuple[str, str]: ...


def is_agent_target(obj: object) -> bool:
    """Return True if obj satisfies the AgentTarget protocol at runtime.

    Checks for ``send_prompt`` and ``new``.
    """
    has_send = callable(getattr(obj, 'send_prompt', None))
    has_new = callable(getattr(obj, 'new', None))
    return has_send and has_new


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


class DirectTargetFactory:
    """Fallback factory that wraps a bare AgentTarget."""

    def __init__(self, target: AgentTarget) -> None:
        self._target = target

    def create_target(self, agent_key: str) -> AgentTarget:
        result = self._target.new()
        if result is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise TypeError(
                f"{type(self._target).__name__}.new() returned None. "
                "It must return a fresh AgentTarget instance."
            )
        return result


class NoopMemoryCleanup:
    """No-op memory cleanup for targets that do not manage memory stores."""

    async def cleanup_memory(self, agent_context: AgentContext, entity_ids: list[str]) -> None:
        logger.debug('Skipping memory cleanup: target does not manage memory stores')


class AgentContextProvider(Protocol):
    """Protocol for retrieving agent context (tools, memory, system prompt)."""

    async def get_agent_context(self, agent_key: str) -> AgentContext:
        """Retrieve full agent context for the given key."""
        ...


class AgentTargetFactory(Protocol):
    """Protocol for creating AgentTarget instances per job.

    The returned target owns its own ``memory_entity_id`` (auto-generated for
    targets with persistent memory, ``None`` otherwise). Callers read the
    attribute off the target rather than passing an ID in.
    """

    def create_target(self, agent_key: str) -> AgentTarget:
        """Create a new AgentTarget for the given agent key."""
        ...


class MemoryCleanup(Protocol):
    """Protocol for cleaning up memory entities created during red teaming."""

    async def cleanup_memory(self, agent_context: AgentContext, entity_ids: list[str]) -> None:
        """Delete memory entities created during a red teaming run."""
        ...


class ErrorMapper(Protocol):
    """Protocol for backend-specific exception normalization."""

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Return normalized (error_code, error_message)."""
        ...


@dataclass(slots=True)
class BackendBundle:
    """Bundle of backend components used by dynamic runtime."""

    name: str
    target_factory: AgentTargetFactory
    context_provider: AgentContextProvider
    memory_cleanup: MemoryCleanup
    error_mapper: ErrorMapper


class DefaultErrorMapper:
    """Fallback error mapper for unknown backends."""

    def map_error(self, exc: Exception) -> tuple[str, str]:
        return 'target_error', f'{type(exc).__name__}: {exc}'


def extract_status_code(exc: Exception) -> int | None:
    """Extract HTTP-like status code from structured exception fields or text."""
    # Structured response.status_code (httpx/openai-like)
    response = getattr(exc, 'response', None)
    status_code = getattr(response, 'status_code', None)
    if isinstance(status_code, int) and 100 <= status_code <= 599:
        return status_code

    # Some SDKs expose status/status_code directly on exception.
    for attr in ('status_code', 'status'):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    # Fallback to regex on raw error text.
    text = str(exc)
    patterns = [
        r'\bstatus(?:_code)?\s*[=:]\s*(\d{3})\b',
        r'\bHTTP\s*(\d{3})\b',
        r'\bcode\s*[=:]\s*(\d{3})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        code = int(match.group(1))
        if 100 <= code <= 599:
            return code
    return None


def extract_provider_error_code(exc: Exception) -> str | None:
    """Extract provider-specific symbolic error code if present."""
    # Common structured fields.
    for attr in ('code', 'error_code', 'type'):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    body = getattr(exc, 'body', None)
    if isinstance(body, dict):
        error = body.get('error') if isinstance(body.get('error'), dict) else body
        for key in ('code', 'type', 'error_code'):
            value = error.get(key) if isinstance(error, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    text = str(exc)
    patterns = [
        r'\b(?:error_)?code\s*[=:]\s*["\']?([a-z0-9_.-]+)["\']?',
        r'\btype\s*[=:]\s*["\']?([a-z0-9_.-]+)["\']?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    return None
