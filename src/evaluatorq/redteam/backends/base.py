"""Backend protocols for dynamic red teaming agent targets.

Defines the abstract interfaces that any agent backend must implement.
The ORQ implementation lives in ``backends.orq``; other backends (HTTP,
LangChain, custom callables) can implement these protocols independently.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from loguru import logger

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import AgentContext, TokenUsage


class AgentTarget(Protocol):
    """Protocol for agent targets that can receive prompts."""

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt and return the response."""
        ...

    def reset_conversation(self) -> None:
        """Reset conversation state for a new attack."""
        ...


class SupportsClone(Protocol):
    """Optional hook for target cloning per parallel job.

    Note: The runner detects this via duck-typing (``getattr``/``callable``),
    not ``isinstance``.  Implementing this protocol is recommended for
    documentation and static analysis but not strictly required.
    """

    def clone(self, memory_entity_id: str | None = None) -> AgentTarget:
        """Return a fresh target instance, optionally with a different memory entity."""
        ...


class SupportsTokenUsage(Protocol):
    """Optional hook for exposing token usage from last target call."""

    def consume_last_token_usage(self) -> TokenUsage | None:
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
    """Optional: target can create per-job instances."""

    def create_target(self, agent_key: str, memory_entity_id: str | None = None) -> AgentTarget: ...


class SupportsMemoryCleanup(Protocol):
    """Optional: target can clean up memory entities."""

    async def cleanup_memory(self, agent_context: AgentContext, entity_ids: list[str]) -> None: ...


class SupportsErrorMapping(Protocol):
    """Optional: target provides custom error classification."""

    def map_error(self, exc: Exception) -> tuple[str, str]: ...


def is_agent_target(obj: object) -> bool:
    """Return True if obj satisfies the AgentTarget protocol at runtime."""
    return callable(getattr(obj, 'send_prompt', None)) and callable(getattr(obj, 'reset_conversation', None))


class DirectTargetFactory:
    """Fallback factory that wraps a bare AgentTarget (no SupportsTargetFactory)."""

    def __init__(self, target: AgentTarget) -> None:
        self._target = target
        clone_attr = getattr(target, 'clone', None)
        self._clone_fn = clone_attr if callable(clone_attr) else None
        if self._clone_fn is None:
            logger.warning(
                f'Target {type(target).__name__} does not implement clone(). '
                'Reusing same instance across parallel jobs may cause race conditions.'
            )

    def create_target(self, agent_key: str, memory_entity_id: str | None = None) -> AgentTarget:
        if self._clone_fn is not None:
            try:
                sig = inspect.signature(self._clone_fn)
                has_memory_param = 'memory_entity_id' in sig.parameters
            except (ValueError, TypeError):
                has_memory_param = False
            if has_memory_param:
                return cast("AgentTarget", self._clone_fn(memory_entity_id=memory_entity_id))
            logger.warning(
                f'{type(self._target).__name__}.clone() does not accept memory_entity_id. '
                'Parallel jobs may share memory state. '
                'Add memory_entity_id: str | None = None to clone() to fix this.'
            )
            return cast("AgentTarget", self._clone_fn())
        return self._target


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
    """Protocol for creating AgentTarget instances per job."""

    def create_target(self, agent_key: str, memory_entity_id: str | None = None) -> AgentTarget:
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
