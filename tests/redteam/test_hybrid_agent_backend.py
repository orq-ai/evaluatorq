"""Tests for HybridAgentBackend composite backend."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.redteam.backends.base import HybridAgentBackend
from evaluatorq.redteam.contracts import AgentContext


def _make_backend(name: str = "stub") -> MagicMock:
    """Return a MagicMock that quacks like a Backend."""
    backend = MagicMock(name=name)
    backend.resolve_context = AsyncMock()
    backend.cleanup_memory = AsyncMock()
    return backend


def test_create_target_prefixes_key_and_delegates_to_exec():
    context_backend = _make_backend("context")
    exec_backend = _make_backend("exec")
    sentinel_target = object()
    exec_backend.create_target.return_value = sentinel_target

    hybrid = HybridAgentBackend(context_backend=context_backend, exec_backend=exec_backend)
    result = hybrid.create_target("my-key")

    exec_backend.create_target.assert_called_once_with("agent/my-key")
    context_backend.create_target.assert_not_called()
    assert result is sentinel_target


@pytest.mark.asyncio
async def test_resolve_context_delegates_bare_key_to_context_backend():
    context_backend = _make_backend("context")
    exec_backend = _make_backend("exec")
    sentinel_ctx = AgentContext(key="my-key")
    context_backend.resolve_context.return_value = sentinel_ctx

    hybrid = HybridAgentBackend(context_backend=context_backend, exec_backend=exec_backend)
    result = await hybrid.resolve_context("my-key")

    context_backend.resolve_context.assert_called_once_with("my-key")
    exec_backend.resolve_context.assert_not_called()
    assert result is sentinel_ctx


@pytest.mark.asyncio
async def test_cleanup_memory_delegates_to_context_backend_not_exec():
    context_backend = _make_backend("context")
    exec_backend = _make_backend("exec")
    ctx = AgentContext(key="my-key")
    entity_ids = ["id-1", "id-2"]

    hybrid = HybridAgentBackend(context_backend=context_backend, exec_backend=exec_backend)
    await hybrid.cleanup_memory(ctx, entity_ids)

    context_backend.cleanup_memory.assert_called_once_with(ctx, entity_ids)
    exec_backend.cleanup_memory.assert_not_called()


def test_map_error_delegates_to_exec_backend():
    context_backend = _make_backend("context")
    exec_backend = _make_backend("exec")
    exec_backend.map_error.return_value = ("exec.error", "something went wrong")

    hybrid = HybridAgentBackend(context_backend=context_backend, exec_backend=exec_backend)
    exc = RuntimeError("boom")
    result = hybrid.map_error(exc)

    exec_backend.map_error.assert_called_once_with(exc)
    context_backend.map_error.assert_not_called()
    assert result == ("exec.error", "something went wrong")


def test_requires_exactly_one_context_source():
    exec_backend = _make_backend("exec")
    # Neither provided.
    with pytest.raises(ValueError, match="exactly one"):
        HybridAgentBackend(exec_backend=exec_backend)
    # Both provided.
    with pytest.raises(ValueError, match="exactly one"):
        HybridAgentBackend(
            context_backend=_make_backend("context"),
            context_backend_factory=lambda: _make_backend("context"),
            exec_backend=exec_backend,
        )


def test_lazy_context_factory_not_called_for_execution_paths():
    """A static agent: run only uses exec — the ORQ SDK context backend (and its
    orq-ai-sdk import) must not be built until context resolution is actually needed."""
    exec_backend = _make_backend("exec")
    exec_backend.map_error.return_value = ("exec.error", "boom")
    factory = MagicMock(side_effect=lambda: _make_backend("context"))

    hybrid = HybridAgentBackend(context_backend_factory=factory, exec_backend=exec_backend)
    # Construction must not build the context backend.
    factory.assert_not_called()
    # Execution paths must not build it either.
    hybrid.create_target("k")
    hybrid.map_error(RuntimeError("x"))
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_lazy_context_factory_built_once_on_first_context_use():
    context_backend = _make_backend("context")
    factory = MagicMock(return_value=context_backend)
    exec_backend = _make_backend("exec")

    hybrid = HybridAgentBackend(context_backend_factory=factory, exec_backend=exec_backend)
    ctx = AgentContext(key="my-key")
    await hybrid.resolve_context("my-key")
    await hybrid.cleanup_memory(ctx, ["id-1"])

    # Built exactly once (cached), and both calls reached the real context backend.
    factory.assert_called_once_with()
    context_backend.resolve_context.assert_called_once_with("my-key")
    context_backend.cleanup_memory.assert_called_once_with(ctx, ["id-1"])
