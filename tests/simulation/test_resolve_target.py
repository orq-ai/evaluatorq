"""Unit tests for the SDK target resolution (string / AgentTarget / callable)."""

from __future__ import annotations

import pytest

from evaluatorq.simulation.api import _resolve_target


@pytest.fixture(autouse=True)
def _orq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The agent path builds the Responses-router backend (no network) that needs a key.
    monkeypatch.setenv("ORQ_API_KEY", "test-key")


def test_agent_prefix_resolves_to_orq_agent() -> None:
    callback, agent, kind = _resolve_target("agent:support")
    assert callback is None
    assert agent is not None
    assert kind == "orq_agent"


def test_bare_string_resolves_to_orq_agent() -> None:
    callback, agent, kind = _resolve_target("support")
    assert callback is None
    assert agent is not None
    assert kind == "orq_agent"


def test_deployment_prefix_resolves_to_orq_deployment() -> None:
    callback, agent, kind = _resolve_target("deployment:support")
    assert callable(callback)
    assert agent is None
    assert kind == "orq_deployment"


def test_callable_resolves_to_callback() -> None:
    async def my_agent(messages):  # noqa: ANN001, ANN202
        return "hi"

    callback, agent, kind = _resolve_target(my_agent)
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
    callback, agent, kind = _resolve_target(stub)
    assert callback is None
    assert agent is stub
    assert kind == "orq_agent"


def test_missing_target_raises() -> None:
    with pytest.raises(ValueError, match="target is required"):
        _resolve_target(None)


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _resolve_target("   ")


def test_non_callable_target_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported target type"):
        _resolve_target(123)  # pyright: ignore[reportArgumentType]


def test_agent_target_gets_memory_entity_id() -> None:
    _, agent, kind = _resolve_target("agent:support-bot", memory_entity_id="sim-e1")
    assert kind == "orq_agent"
    assert agent is not None
    assert agent.memory_entity_id == "sim-e1"


def test_agent_target_memory_entity_id_defaults_to_minted_id() -> None:
    # The backend mints a fresh per-target memory id when none is given, so
    # memory-backed agents work out of the box; an explicit id (previous test)
    # overrides it. Two resolves must not share a memory scope.
    _, agent_a, _ = _resolve_target("agent:support-bot")
    _, agent_b, _ = _resolve_target("agent:support-bot")
    assert agent_a is not None
    assert agent_b is not None
    assert agent_a.memory_entity_id
    assert agent_a.memory_entity_id.startswith("red-team-")
    assert agent_a.memory_entity_id != agent_b.memory_entity_id


def test_blank_memory_entity_id_rejected() -> None:
    # A blank id would be silently dropped by the target's falsy check and
    # reproduce the exact 400 the parameter exists to prevent.
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="blank"):
            _resolve_target("agent:support-bot", memory_entity_id=blank)


def test_memory_entity_id_rejected_for_deployment_target() -> None:
    with pytest.raises(ValueError, match="memory_entity_id"):
        _resolve_target("deployment:my-deploy", memory_entity_id="sim-e1")


def test_memory_entity_id_rejected_for_instance_target() -> None:
    from evaluatorq.contracts import AgentResponse, AgentTarget, Message

    class _StubTarget(AgentTarget):
        async def respond(self, messages: list[Message]) -> AgentResponse:
            return AgentResponse(text="hi")

        def new(self) -> "_StubTarget":
            return _StubTarget()

    with pytest.raises(ValueError, match="memory_entity_id"):
        _resolve_target(_StubTarget(), memory_entity_id="sim-e1")
