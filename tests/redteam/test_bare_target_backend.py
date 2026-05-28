"""Tests for BareTargetBackend adapter."""
from __future__ import annotations

import pytest

from evaluatorq.redteam.backends.base import AgentTarget, BareTargetBackend
from evaluatorq.redteam.contracts import AgentContext, AgentResponse


class _StubTarget(AgentTarget):
    def __init__(self, memory_entity_id: str | None = None) -> None:
        super().__init__(memory_entity_id=memory_entity_id)

    async def send_prompt(self, prompt: str) -> AgentResponse:
        return AgentResponse(text="ok")

    def new(self) -> "_StubTarget":
        return _StubTarget()


class _TargetWithCleanup(_StubTarget):
    def __init__(self):
        super().__init__()
        self.cleaned: list[str] = []

    async def cleanup_memory(self, ctx, entity_ids):
        self.cleaned.extend(entity_ids)


class _TargetWithMapping(_StubTarget):
    def map_error(self, exc):
        return ("byo.error", str(exc))


@pytest.mark.asyncio
async def test_bare_backend_delegates_cleanup_when_target_supports_it():
    target = _TargetWithCleanup()
    backend = BareTargetBackend(target)
    await backend.cleanup_memory(AgentContext(key="x"), ["a", "b"])
    assert target.cleaned == ["a", "b"]


@pytest.mark.asyncio
async def test_bare_backend_cleanup_noop_when_target_lacks_it():
    backend = BareTargetBackend(_StubTarget())
    # Must not raise.
    await backend.cleanup_memory(AgentContext(key="x"), ["a"])


def test_bare_backend_delegates_map_error_when_target_supports_it():
    target = _TargetWithMapping()
    backend = BareTargetBackend(target)
    code, _ = backend.map_error(RuntimeError("boom"))
    assert code == "byo.error"


def test_bare_backend_create_target_returns_target_new():
    target = _StubTarget()
    backend = BareTargetBackend(target)
    fresh = backend.create_target("ignored-agent-key")
    assert isinstance(fresh, _StubTarget)
    assert fresh is not target
