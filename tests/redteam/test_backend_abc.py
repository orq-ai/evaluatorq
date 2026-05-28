"""Tests for Backend ABC in redteam.backends.base."""
from __future__ import annotations

import pytest

from evaluatorq.redteam.backends.base import Backend


class _MinimalBackend(Backend):
    def create_target(self, agent_key):
        raise NotImplementedError

    async def cleanup_memory(self, ctx, entity_ids):
        return None


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        Backend("x")  # pyright: ignore[reportAbstractUsage]


def test_backend_subclass_sets_name():
    b = _MinimalBackend("orq")
    assert b.name == "orq"


def test_default_map_error_returns_target_error_tuple():
    b = _MinimalBackend("orq")
    code, msg = b.map_error(RuntimeError("boom"))
    assert code == "target_error"
    assert "RuntimeError" in msg
    assert "boom" in msg


def test_orq_backend_map_error_includes_status_code():
    from evaluatorq.redteam.backends.orq import ORQBackend

    class _HTTPError(Exception):
        def __init__(self):
            super().__init__("boom")
            self.status_code = 429

    backend = ORQBackend(orq_client=object(), timeout_ms=1000)
    code, _ = backend.map_error(_HTTPError())
    assert code == "orq.http.429"


def test_openai_backend_map_error_returns_openai_prefix():
    from unittest.mock import MagicMock

    from evaluatorq.redteam.backends.openai import OpenAIBackend

    # Pass MagicMock as client — constructor's ``client or create_async_llm_client()``
    # would otherwise trigger CredentialError when no API key is set in the test env.
    backend = OpenAIBackend(client=MagicMock(), system_prompt=None)
    code, _ = backend.map_error(TimeoutError("slow"))
    assert code.startswith("openai.")


class _CountingBackend(Backend):
    """Backend whose create_target is counted, returning a target with a fixed context."""

    def __init__(self):
        super().__init__(name="counting")
        self.create_calls = 0

    def create_target(self, agent_key):
        from unittest.mock import AsyncMock, MagicMock

        from evaluatorq.redteam.contracts import AgentContext

        self.create_calls += 1
        target = MagicMock()
        target.get_agent_context = AsyncMock(return_value=AgentContext(key=agent_key))
        return target

    async def cleanup_memory(self, ctx, entity_ids):
        return None


@pytest.mark.asyncio
async def test_resolve_context_caches_per_key():
    """Default resolve_context probes once per key, then serves from cache."""
    backend = _CountingBackend()

    first = await backend.resolve_context("agent-a")
    second = await backend.resolve_context("agent-a")

    assert first is second, "cache hit must return the same AgentContext instance"
    assert backend.create_calls == 1, "second call must not re-probe a new target"


@pytest.mark.asyncio
async def test_resolve_context_caches_distinct_keys_independently():
    """Distinct agent keys are probed and cached separately."""
    backend = _CountingBackend()

    ctx_a = await backend.resolve_context("agent-a")
    ctx_b = await backend.resolve_context("agent-b")

    assert ctx_a.key == "agent-a"
    assert ctx_b.key == "agent-b"
    assert backend.create_calls == 2
