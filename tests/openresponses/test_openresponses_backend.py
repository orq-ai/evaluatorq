"""Tests for OpenResponsesAgentTarget.

Verifies the live wire shape: when a red team attack is sent through this
target, the payload posted to ``client.responses.create`` matches the
RES-540 ticket spec, multi-turn state threads correctly, and the resulting
``AgentResponse`` carries the parsed output items.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import TextOutputItem, ToolCallOutputItem
from evaluatorq.redteam.backends.base import is_agent_target
from evaluatorq.redteam.backends.openresponses import (
    OpenResponsesAgentTarget,
    OpenResponsesTargetFactory,
)


def _fake_response(
    *,
    text: str = "agent reply",
    response_id: str | None = "resp_1",
    model: str = "agent-id",
    status: str = "completed",
    input_tokens: int = 5,
    output_tokens: int = 3,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a SimpleNamespace mimicking a ``client.responses.create`` return value."""
    output: list[Any] = []
    if text:
        output.append(SimpleNamespace(
            type="message",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text=text)],
        ))
    for tc in tool_calls or []:
        output.append(SimpleNamespace(
            type="function_call",
            name=tc["name"],
            call_id=tc["call_id"],
            arguments=tc["arguments"],
            result=None,
            id=tc["call_id"],
        ))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        id=response_id,
        model=model,
        status=status,
        output=output,
        usage=usage,
    )


def _make_target(
    *,
    response: Any | None = None,
    responses: list[Any] | None = None,
    **kwargs: Any,
) -> tuple[OpenResponsesAgentTarget, MagicMock]:
    """Construct a target with a mocked ``client.responses.create``."""
    create_mock = AsyncMock()
    if responses is not None:
        create_mock.side_effect = responses
    else:
        create_mock.return_value = response if response is not None else _fake_response()
    client = MagicMock()
    client.responses = SimpleNamespace(create=create_mock)
    target = OpenResponsesAgentTarget(agent_id="agent-id", client=client, **kwargs)
    return target, create_mock


class TestProtocolConformance:
    def test_satisfies_agent_target_protocol(self):
        target, _ = _make_target()
        assert is_agent_target(target)

    def test_memory_entity_id_is_none(self):
        target, _ = _make_target()
        assert target.memory_entity_id is None

    def test_new_returns_independent_instance(self):
        target, _ = _make_target()
        fresh = target.new()
        assert fresh is not target
        assert fresh.agent_id == target.agent_id


class TestWirePayloadShape:
    @pytest.mark.asyncio
    async def test_initial_call_matches_ticket_spec(self):
        target, create = _make_target()
        await target.send_prompt("adversarial prompt here")

        kwargs = create.call_args.kwargs
        # Ticket spec exact shape:
        # { "model": "agent-id",
        #   "input": [{"role": "user", "content": "adversarial prompt here"}] }
        assert kwargs["model"] == "agent-id"
        assert kwargs["input"] == [{"role": "user", "content": "adversarial prompt here"}]

    @pytest.mark.asyncio
    async def test_instructions_passed_through_when_set(self):
        target, create = _make_target(instructions="be helpful")
        await target.send_prompt("attack")
        assert create.call_args.kwargs["instructions"] == "be helpful"

    @pytest.mark.asyncio
    async def test_tools_passed_through_when_set(self):
        tools = [{"type": "function", "name": "search", "parameters": {}}]
        target, create = _make_target(tools=tools)
        await target.send_prompt("attack")
        assert create.call_args.kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_max_output_tokens_passed_through(self):
        target, create = _make_target(max_tokens=512)
        await target.send_prompt("attack")
        assert create.call_args.kwargs["max_output_tokens"] == 512


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_parses_text_output_into_agent_response(self):
        target, _ = _make_target(response=_fake_response(text="agent reply"))
        result = await target.send_prompt("attack")
        assert result.text == "agent reply"

    @pytest.mark.asyncio
    async def test_parses_function_calls(self):
        target, _ = _make_target(response=_fake_response(
            text="checking",
            tool_calls=[{"name": "search", "call_id": "call_1", "arguments": '{"q":"x"}'}],
        ))
        result = await target.send_prompt("attack")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].call_id == "call_1"

    @pytest.mark.asyncio
    async def test_carries_response_id_model_finish_reason(self):
        target, _ = _make_target(response=_fake_response(
            response_id="resp_42", model="agent-id", status="completed",
        ))
        result = await target.send_prompt("attack")
        assert result.response_id == "resp_42"
        assert result.model == "agent-id"
        assert result.finish_reason == "completed"

    @pytest.mark.asyncio
    async def test_usage_translated_into_redteam_token_usage(self):
        target, _ = _make_target(response=_fake_response(input_tokens=12, output_tokens=8))
        result = await target.send_prompt("attack")
        assert result.usage is not None
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 8
        assert result.usage.calls == 1


class TestMultiTurnThreading:
    @pytest.mark.asyncio
    async def test_server_threading_passes_previous_response_id(self):
        target, create = _make_target(responses=[
            _fake_response(text="first reply", response_id="resp_1"),
            _fake_response(text="second reply", response_id="resp_2"),
        ])

        await target.send_prompt("first attack")
        await target.send_prompt("follow-up attack")

        # Second call sends only the new user turn and threads via previous_response_id.
        second_kwargs = create.call_args_list[1].kwargs
        assert second_kwargs["previous_response_id"] == "resp_1"
        assert second_kwargs["input"] == [
            {"role": "user", "content": "follow-up attack"},
        ]

    @pytest.mark.asyncio
    async def test_client_threading_resends_full_input_array(self):
        target, create = _make_target(
            responses=[
                _fake_response(text="first reply", response_id="resp_1"),
                _fake_response(text="second reply", response_id="resp_2"),
            ],
            use_server_threading=False,
        )

        await target.send_prompt("first attack")
        await target.send_prompt("follow-up attack")

        second_kwargs = create.call_args_list[1].kwargs
        assert "previous_response_id" not in second_kwargs
        assert second_kwargs["input"] == [
            {"role": "user", "content": "first attack"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "follow-up attack"},
        ]

    @pytest.mark.asyncio
    async def test_missing_response_id_falls_back_to_client_threading(self):
        target, create = _make_target(responses=[
            _fake_response(text="first reply", response_id=None),
            _fake_response(text="second reply", response_id="resp_2"),
        ])

        await target.send_prompt("first attack")
        await target.send_prompt("follow-up attack")

        second_kwargs = create.call_args_list[1].kwargs
        assert "previous_response_id" not in second_kwargs
        # Client-side fallback should now carry both turns.
        assert second_kwargs["input"][0] == {"role": "user", "content": "first attack"}
        assert second_kwargs["input"][1] == {"role": "assistant", "content": "first reply"}
        assert second_kwargs["input"][-1] == {"role": "user", "content": "follow-up attack"}

    @pytest.mark.asyncio
    async def test_new_resets_conversation_state(self):
        target, create = _make_target(responses=[
            _fake_response(response_id="resp_1"),
            _fake_response(response_id="resp_other"),
        ])
        await target.send_prompt("attack")
        fresh = target.new()
        # fresh shares the client but has no carried-over threading.
        await fresh.send_prompt("new attack")
        second_kwargs = create.call_args_list[1].kwargs
        assert "previous_response_id" not in second_kwargs


class TestErrorMapping:
    def test_timeout_maps_to_openresponses_timeout(self):
        import asyncio as _asyncio
        target, _ = _make_target()
        code, _ = target.map_error(_asyncio.TimeoutError())
        assert code == "openresponses.timeout"

    def test_unknown_exception_maps_to_unknown(self):
        target, _ = _make_target()
        code, _ = target.map_error(RuntimeError("boom"))
        assert code == "openresponses.unknown"


class TestFactory:
    def test_factory_creates_per_job_targets(self):
        client = MagicMock()
        factory = OpenResponsesTargetFactory(client, instructions="be safe")
        t1 = factory.create_target("agent-a")
        t2 = factory.create_target("agent-b")
        assert t1 is not t2
        assert t1.agent_id == "agent-a"
        assert t2.agent_id == "agent-b"
        assert t1.instructions == "be safe"
