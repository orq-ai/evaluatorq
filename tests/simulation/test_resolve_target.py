"""Unit tests for the SDK target resolution (string / AgentTarget / callable)."""

from __future__ import annotations

import pytest

from evaluatorq.simulation.api import _resolve_target


@pytest.fixture(autouse=True)
def _orq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The agent path builds the Responses-router backend (no network) that needs a key.
    monkeypatch.setenv("ORQ_API_KEY", "test-key")


def test_agent_prefix_resolves_to_orq_agent() -> None:
    callback, agent, kind = _resolve_target("agent:support", None)
    assert callback is None
    assert agent is not None
    assert kind == "orq_agent"


def test_bare_string_resolves_to_orq_agent() -> None:
    callback, agent, kind = _resolve_target("support", None)
    assert callback is None
    assert agent is not None
    assert kind == "orq_agent"


def test_deployment_prefix_resolves_to_orq_deployment() -> None:
    callback, agent, kind = _resolve_target("deployment:support", None)
    assert callable(callback)
    assert agent is None
    assert kind == "orq_deployment"


def test_callable_resolves_to_callback() -> None:
    async def my_agent(messages):  # noqa: ANN001, ANN202
        return "hi"

    callback, agent, kind = _resolve_target(my_agent, None)
    assert callback is my_agent
    assert agent is None
    assert kind is None  # -> 'callback' in the save block


def test_agent_target_instance_resolves_to_orq_agent() -> None:
    from evaluatorq.contracts import AgentResponse, AgentTarget, Message

    class _StubTarget(AgentTarget):
        async def respond(self, messages: list[Message]) -> AgentResponse:
            return AgentResponse(text="hi")

        def new(self) -> "_StubTarget":
            return _StubTarget()

    stub = _StubTarget()
    callback, agent, kind = _resolve_target(stub, None)
    assert callback is None
    assert agent is stub
    assert kind == "orq_agent"


def test_missing_target_raises() -> None:
    with pytest.raises(ValueError, match="target is required"):
        _resolve_target(None, None)


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _resolve_target("   ", None)


def test_non_callable_target_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported target type"):
        _resolve_target(123, None)  # type: ignore[arg-type]
