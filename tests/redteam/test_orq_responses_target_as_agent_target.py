"""Tests for OrqResponsesTarget conformance with the AgentTarget ABC.

After RES-877 Task 9:
- ``respond(messages)`` is the sole response method; send_prompt shim removed
- ``OrqResponsesTarget`` is fully stateless — no ``_previous_response_id`` or
  ``get_usage`` invariants
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, LLMCallConfig, Message
from evaluatorq.openresponses.target import OrqResponsesTarget


def _make_client() -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


def _make_response(text: str = "all good") -> MagicMock:
    part = MagicMock()
    part.type = "output_text"
    part.text = text
    msg_item = MagicMock()
    msg_item.type = "message"
    msg_item.content = [part]
    usage = MagicMock()
    usage.input_tokens = 5
    usage.output_tokens = 3
    response = MagicMock()
    response.id = "resp-1"
    response.usage = usage
    response.output = [msg_item]
    return response


def _make_target() -> OrqResponsesTarget:
    client = _make_client()
    client.responses.create = AsyncMock(return_value=_make_response())
    return OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)


class TestAgentTargetConformance:
    def test_is_agent_target_instance(self):
        assert isinstance(_make_target(), AgentTarget)

    def test_memory_entity_id_default_none(self):
        assert _make_target().memory_entity_id is None

    def test_memory_entity_id_settable(self):
        client = _make_client()
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"), client=client, memory_entity_id="x-1"
        )
        assert target.memory_entity_id == "x-1"


class TestRespond:
    @pytest.mark.asyncio
    async def test_respond_returns_agent_response(self):
        target = _make_target()
        result = await target.respond([Message(role="user", content="hi")])
        assert isinstance(result, AgentResponse)
        assert result.text == "all good"


class TestRespondIsStateless:
    @pytest.mark.asyncio
    async def test_consecutive_respond_calls_pass_messages_as_sent(self):
        """respond is stateless: each call's input is exactly what the caller passed.

        No previous_response_id threading, no accumulation on self.
        """
        client = _make_client()
        client.responses.create = AsyncMock(
            side_effect=[_make_response("r1"), _make_response("r2")]
        )
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)

        await target.respond([Message(role="user", content="turn1")])
        await target.respond([Message(role="user", content="turn2")])

        call1_kwargs = client.responses.create.await_args_list[0].kwargs
        call2_kwargs = client.responses.create.await_args_list[1].kwargs
        assert "previous_response_id" not in call1_kwargs
        assert "previous_response_id" not in call2_kwargs
        assert call1_kwargs["input"] == [{"role": "user", "content": "turn1"}]
        assert call2_kwargs["input"] == [{"role": "user", "content": "turn2"}]

    @pytest.mark.asyncio
    async def test_respond_routes_single_user_message(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)

        await target.respond([Message(role="user", content="attack prompt")])

        call_kwargs = client.responses.create.await_args_list[-1].kwargs
        assert call_kwargs["input"] == [{"role": "user", "content": "attack prompt"}]
        assert "previous_response_id" not in call_kwargs


class TestNew:
    def test_new_returns_different_instance(self):
        target = _make_target()
        assert target.new() is not target

    def test_new_memory_entity_id_is_none(self):
        target = _make_target()
        assert target.new().memory_entity_id is None

    def test_new_propagates_injected_client(self):
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)
        assert target.new()._client is client

    def test_new_preserves_constructor_seeded_memory_entity(self):
        """A seeded id must survive cloning (the sim --memory-entity path);
        mirrors ORQAgentTarget.new() semantics."""
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"), client=_make_client(), memory_entity_id="seeded-entity"
        )
        assert target.new().memory_entity_id == "seeded-entity"

    def test_new_preserves_assignment_seeded_memory_entity(self):
        target = _make_target()
        target.memory_entity_id = "seeded-entity"
        assert target.new().memory_entity_id == "seeded-entity"

    def test_seeded_id_survives_grandchild_clones(self):
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"), client=_make_client(), memory_entity_id="seeded-entity"
        )
        assert target.new().new().memory_entity_id == "seeded-entity"

    def test_unseeding_via_assignment_reverts_to_none_clones(self):
        target = OrqResponsesTarget(
            LLMCallConfig(model="gpt-4o"), client=_make_client(), memory_entity_id="seeded-entity"
        )
        target.memory_entity_id = None
        assert target.new().memory_entity_id is None


class TestHybridBackendMintStaysUnseeded:
    """The HybridAgentBackend auto-mint is not a user seed: clones must
    re-mint it (parallel-job isolation), while a later explicit assignment
    seeds and survives clones."""

    def _hybrid_target(self, monkeypatch):
        from evaluatorq.redteam.backends.registry import make_agent_backend
        from evaluatorq.redteam.contracts import LLMConfig, TargetConfig

        monkeypatch.setenv("ORQ_API_KEY", "test-key")
        backend = make_agent_backend(
            target_config=TargetConfig(system_prompt=None), pipeline_config=LLMConfig()
        )
        return backend.create_target("my-agent")

    def test_auto_minted_id_re_mints_per_clone(self, monkeypatch):
        target = self._hybrid_target(monkeypatch)
        assert target.memory_entity_id is not None
        assert target.memory_entity_id.startswith("red-team-")
        clone = target.new()
        assert clone.memory_entity_id is not None
        assert clone.memory_entity_id != target.memory_entity_id

    def test_explicit_seed_after_create_survives_clone(self, monkeypatch):
        target = self._hybrid_target(monkeypatch)
        target.memory_entity_id = "seeded-entity"
        assert target.new().memory_entity_id == "seeded-entity"
