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
from evaluatorq.simulation.target import OrqResponsesTarget


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

    def test_send_prompt_shim_removed(self):
        """send_prompt shim was removed in RES-877 Task 9."""
        from evaluatorq.contracts import AgentTarget
        assert not hasattr(AgentTarget, "send_prompt")


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
