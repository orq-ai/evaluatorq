"""Unit tests for OpenAI Agents SDK red teaming target."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("agents")

from evaluatorq.contracts import FunctionCall, Message, StrategyToolCall, ToolCallOutputItem  # noqa: E402
from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget  # noqa: E402
from evaluatorq.integrations.openai_agents_integration.target import (  # noqa: E402
    _message_to_responses_input_items,
)
from evaluatorq.redteam.contracts import AgentResponse, TokenUsage  # noqa: E402


def _mock_runner_and_result(output: str = "response") -> tuple[MagicMock, MagicMock]:
    result = MagicMock()
    result.final_output = output
    result.to_input_list.return_value = [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": output},
    ]
    # By default, no context_wrapper so usage=None
    del result.context_wrapper
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    return runner, result


def _mock_runner_with_usage(
    output: str = "response",
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int = 15,
) -> tuple[MagicMock, MagicMock]:
    """Build a mock runner where result.context_wrapper.usage carries token counts."""
    agent_usage = MagicMock()
    agent_usage.input_tokens = input_tokens
    agent_usage.output_tokens = output_tokens
    agent_usage.total_tokens = total_tokens

    context_wrapper = MagicMock()
    context_wrapper.usage = agent_usage

    result = MagicMock()
    result.final_output = output
    result.context_wrapper = context_wrapper
    result.to_input_list.return_value = [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": output},
    ]

    runner = MagicMock()
    runner.run = AsyncMock(return_value=result)
    return runner, result


class TestOpenAIAgentTarget:
    @pytest.mark.asyncio
    async def test_respond_returns_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner, _ = _mock_runner_and_result("hello back")
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        response = await target.respond([Message(role="user", content="hello")])
        assert isinstance(response, AgentResponse)
        assert response.text == "hello back"

    @pytest.mark.asyncio
    async def test_respond_passes_messages_as_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner, _ = _mock_runner_and_result()
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.respond([Message(role="user", content="first")])

        call_args = runner.run.call_args
        input_data = call_args[0][1]
        assert isinstance(input_data, list)
        assert input_data[0] == {"role": "user", "content": "first"}

    @pytest.mark.asyncio
    async def test_respond_maps_tool_calls_to_responses_input_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A replayed transcript with tool calls becomes Responses-API input items
        (function_call / function_call_output), not flattened chat messages."""
        runner, _ = _mock_runner_and_result()
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.respond([
            Message(role="user", content="q1"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="lookup", arguments='{"q":"x"}'))],
            ),
            Message(role="tool", tool_call_id="c1", name="lookup", content="found"),
            Message(role="user", content="q2"),
        ])

        input_data = runner.run.call_args[0][1]
        fc = next(i for i in input_data if isinstance(i, dict) and i.get("type") == "function_call")
        assert fc["call_id"] == "c1"
        assert fc["name"] == "lookup"
        assert fc["arguments"] == '{"q":"x"}'
        fco = next(i for i in input_data if isinstance(i, dict) and i.get("type") == "function_call_output")
        assert fco["call_id"] == "c1"
        assert fco["output"] == "found"

    def test_responses_input_items_round_trip_through_build_response(self) -> None:
        """Inverse mapper output is consumable by _build_response, and the
        function_call/function_call_output pair shares one call_id (the SDK
        requirement for pairing a call with its result on replay)."""
        items = _message_to_responses_input_items(
            Message(
                role="assistant",
                content=None,
                tool_calls=[StrategyToolCall(id="call_1", function=FunctionCall(name="lookup", arguments='{"q":"x"}'))],
            )
        )
        items += _message_to_responses_input_items(Message(role="tool", tool_call_id="call_1", content="found"))

        fc = next(i for i in items if i["type"] == "function_call")
        fco = next(i for i in items if i["type"] == "function_call_output")
        assert fc["call_id"] == fco["call_id"] == "call_1"

        target = OpenAIAgentTarget(MagicMock())
        result = MagicMock()
        result.final_output = "done"
        resp = target._build_response(items, result)
        tool_calls = [o for o in resp.output if isinstance(o, ToolCallOutputItem)]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "lookup"
        assert tool_calls[0].arguments == '{"q":"x"}'

    @pytest.mark.asyncio
    async def test_no_warning_when_input_echoed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When to_input_list() echoes our input at the front, the slice is sound — no warning."""
        result = MagicMock()
        result.final_output = "resp"
        result.to_input_list.return_value = [
            {"role": "user", "content": "hi"},  # echoes the rendered input 1:1
            {"role": "assistant", "content": "resp"},
        ]
        del result.context_wrapper
        runner = MagicMock()
        runner.run = AsyncMock(return_value=result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with caplog.at_level("WARNING"):
            resp = await target.respond([Message(role="user", content="hi")])
        assert resp.text == "resp"
        assert "no longer echoes" not in caplog.text

    @pytest.mark.asyncio
    async def test_warns_when_input_not_echoed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the SDK stops echoing input 1:1 at the front, the slice may misalign —
        respond() must warn loudly (not corrupt output silently) and still return a response."""
        result = MagicMock()
        result.final_output = "resp"
        result.to_input_list.return_value = [
            {"role": "system", "content": "normalized"},  # front differs from sent input
            {"role": "assistant", "content": "resp"},
        ]
        del result.context_wrapper
        runner = MagicMock()
        runner.run = AsyncMock(return_value=result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with caplog.at_level("WARNING"):
            resp = await target.respond([Message(role="user", content="hi")])
        assert isinstance(resp, AgentResponse)
        assert "no longer echoes" in caplog.text

    def test_new_returns_independent_instance(self) -> None:
        agent = MagicMock()
        target = OpenAIAgentTarget(agent, run_kwargs={"max_turns": 5})
        fresh = target.new()
        assert fresh is not target
        assert fresh._agent is agent
        assert fresh._run_kwargs is not target._run_kwargs
        assert fresh._run_kwargs == {"max_turns": 5}

    @pytest.mark.asyncio
    async def test_raises_when_final_output_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = MagicMock()
        result.final_output = None
        result.to_input_list.return_value = []
        runner = MagicMock()
        runner.run = AsyncMock(return_value=result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with pytest.raises(ValueError, match="final_output=None"):
            await target.respond([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_runner_error_is_wrapped_with_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = MagicMock()
        runner.run = AsyncMock(side_effect=RuntimeError("model overloaded"))
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with pytest.raises(RuntimeError, match="OpenAIAgentTarget: Runner.run\\(\\) raised"):
            await target.respond([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_runner_timeout_is_not_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = MagicMock()
        runner.run = AsyncMock(side_effect=asyncio.TimeoutError)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with pytest.raises(asyncio.TimeoutError):
            await target.respond([Message(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_extracts_responses_format_function_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = MagicMock()
        result.final_output = "looked it up"
        result.to_input_list.return_value = [
            {"role": "user", "content": "find docs"},
            {
                "type": "function_call",
                "name": "search_docs",
                "arguments": '{"query": "tool calls"}',
            },
            {"role": "assistant", "content": "looked it up"},
        ]
        runner = MagicMock()
        runner.run = AsyncMock(return_value=result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        response = await target.respond([Message(role="user", content="find docs")])

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments_dict == {"query": "tool calls"}

    @pytest.mark.asyncio
    async def test_responses_format_invalid_json_arguments_are_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = MagicMock()
        result.final_output = "done"
        result.to_input_list.return_value = [
            {"role": "user", "content": "run"},
            {"type": "function_call", "name": "run_tool", "arguments": "not-json"},
        ]
        runner = MagicMock()
        runner.run = AsyncMock(return_value=result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        response = await target.respond([Message(role="user", content="run")])

        assert response.tool_calls[0].arguments_dict == {"raw": "not-json"}

    @pytest.mark.asyncio
    async def test_get_agent_context_roundtrips_fields(self) -> None:
        agent = MagicMock()
        agent.name = "support-bot"
        agent.instructions = "Be helpful."
        agent.model = "gpt-4o-mini"
        tool_a = MagicMock()
        tool_a.name = "search_docs"
        tool_a.description = "Search the docs."
        tool_a.params_json_schema = {"type": "object", "properties": {}}
        tool_b = MagicMock()
        tool_b.name = "create_ticket"
        tool_b.description = None
        tool_b.params_json_schema = None
        agent.tools = [tool_a, tool_b]

        target = OpenAIAgentTarget(agent)
        ctx = await target.get_agent_context()

        assert ctx.key == "support-bot"
        assert ctx.system_prompt == "Be helpful."
        assert ctx.model == "gpt-4o-mini"
        assert {t.name for t in ctx.tools} == {"search_docs", "create_ticket"}
        assert ctx.memory_stores == []

    @pytest.mark.asyncio
    async def test_get_agent_context_handles_object_model(self) -> None:
        agent = MagicMock()
        agent.name = "bot"
        agent.instructions = None
        model_obj = MagicMock()
        model_obj.model = "gpt-4o"
        agent.model = model_obj
        agent.tools = []

        target = OpenAIAgentTarget(agent)
        ctx = await target.get_agent_context()
        assert ctx.model == "gpt-4o"
        assert ctx.tools == []


class TestOpenAIAgentTargetUsage:
    @pytest.mark.asyncio
    async def test_usage_populated_from_context_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When result.context_wrapper.usage has token counts, AgentResponse.usage is populated."""
        runner, _ = _mock_runner_with_usage(
            output="hello back",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.respond([Message(role="user", content="hello")])

        assert result.text == "hello back"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_usage_none_when_context_wrapper_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When result has no context_wrapper, AgentResponse.usage is None (graceful fallback)."""
        runner, _ = _mock_runner_and_result("reply")
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.respond([Message(role="user", content="hi")])

        assert result.text == "reply"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_usage_none_when_usage_attr_absent_on_context_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When context_wrapper exists but has no 'usage', AgentResponse.usage is None."""
        ctx = MagicMock(spec=[])  # spec=[] means no attributes by default

        sdk_result = MagicMock()
        sdk_result.final_output = "hello"
        sdk_result.context_wrapper = ctx
        sdk_result.to_input_list.return_value = []

        runner = MagicMock()
        runner.run = AsyncMock(return_value=sdk_result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.respond([Message(role="user", content="hi")])

        assert result.usage is None

    @pytest.mark.asyncio
    async def test_usage_total_tokens_computed_from_parts_when_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When total_tokens is 0, it falls back to prompt + completion sum."""
        runner, _ = _mock_runner_with_usage(
            output="ok",
            input_tokens=8,
            output_tokens=3,
            total_tokens=0,
        )
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.respond([Message(role="user", content="hi")])

        assert result.usage is not None
        assert result.usage.total_tokens == 11  # 8 + 3
