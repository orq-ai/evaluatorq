"""Tests for OrqResponsesTarget conformance with the redteam AgentTarget protocol.

Verifies:
- is_agent_target() returns True for OrqResponsesTarget
- memory_entity_id attribute exists and defaults to None
- send_prompt is callable and returns AgentResponse
- new() returns a different object
- new() fresh instance has previous_response_id == None
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig
from evaluatorq.redteam.backends.base import is_agent_target
from evaluatorq.simulation.target import OrqResponsesTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> MagicMock:
    """Return a mock AsyncOpenAI client with a stub responses.create."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


def _make_response(text: str = "all good") -> MagicMock:
    """Build a minimal mock Responses API response object."""
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
    response.id = "resp-agent-target-test"
    response.usage = usage
    response.output = [msg_item]
    return response


def _make_target() -> OrqResponsesTarget:
    client = _make_client()
    client.responses.create = AsyncMock(return_value=_make_response())
    config = LLMCallConfig(model="gpt-4o")
    return OrqResponsesTarget(config, client=client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIsAgentTarget:
    def test_is_agent_target_returns_true(self):
        """is_agent_target() must return True for OrqResponsesTarget."""
        target = _make_target()
        assert is_agent_target(target) is True

    def test_has_send_prompt_callable(self):
        """send_prompt attribute must exist and be callable."""
        target = _make_target()
        assert callable(getattr(target, "send_prompt", None))

    def test_has_new_callable(self):
        """new attribute must exist and be callable."""
        target = _make_target()
        assert callable(getattr(target, "new", None))


class TestMemoryEntityId:
    def test_memory_entity_id_attribute_exists(self):
        """memory_entity_id attribute must exist (AgentTarget protocol field)."""
        target = _make_target()
        assert hasattr(target, "memory_entity_id")

    def test_memory_entity_id_is_none_by_default(self):
        """memory_entity_id defaults to None when not explicitly set."""
        target = _make_target()
        assert target.memory_entity_id is None

    def test_memory_entity_id_can_be_set(self):
        """memory_entity_id can be set to a non-None value via constructor."""
        client = _make_client()
        config = LLMCallConfig(model="gpt-4o")
        target = OrqResponsesTarget(config, client=client, memory_entity_id="entity-42")
        assert target.memory_entity_id == "entity-42"


class TestSendPrompt:
    @pytest.mark.asyncio
    async def test_send_prompt_is_awaitable(self):
        """send_prompt must be awaitable (coroutine function)."""
        import asyncio

        target = _make_target()
        result = target.send_prompt("test prompt")
        assert asyncio.iscoroutine(result)
        await result  # exhaust the coroutine to avoid ResourceWarning

    @pytest.mark.asyncio
    async def test_send_prompt_returns_agent_response(self):
        """send_prompt must return an AgentResponse instance."""
        target = _make_target()
        result = await target.send_prompt("hello")
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_send_prompt_returns_text(self):
        """send_prompt result has the expected text content."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response(text="safe response"))
        config = LLMCallConfig(model="gpt-4o")
        target = OrqResponsesTarget(config, client=client)

        result = await target.send_prompt("attack prompt")
        assert result.text == "safe response"


class TestNew:
    def test_new_returns_different_object(self):
        """new() must return a new instance, not the same object."""
        target = _make_target()
        fresh = target.new()
        assert fresh is not target

    def test_new_returns_orq_responses_target(self):
        """new() must return an OrqResponsesTarget."""
        target = _make_target()
        fresh = target.new()
        assert isinstance(fresh, OrqResponsesTarget)

    def test_new_result_is_also_agent_target(self):
        """The instance returned by new() must also satisfy is_agent_target."""
        target = _make_target()
        fresh = target.new()
        assert is_agent_target(fresh) is True

    def test_new_fresh_instance_has_previous_response_id_none(self):
        """new() fresh instance must have _previous_response_id == None."""
        target = _make_target()

        # Simulate setting state on the original
        target._previous_response_id = "resp-existing"

        fresh = target.new()
        assert fresh._previous_response_id is None

    def test_new_fresh_memory_entity_id_is_none(self):
        """new() propagates memory_entity_id (None by default)."""
        target = _make_target()
        fresh = target.new()
        assert fresh.memory_entity_id is None

    def test_new_propagates_injected_client(self):
        """new() propagates the injected client to the fresh instance."""
        client = _make_client()
        client.responses.create = AsyncMock(return_value=_make_response())
        config = LLMCallConfig(model="gpt-4o")
        target = OrqResponsesTarget(config, client=client)

        fresh = target.new()
        assert fresh._client is client
