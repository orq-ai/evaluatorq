"""Unit tests for OpenAI Agents SDK red teaming target."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("agents")

from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget  # noqa: E402
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
    async def test_send_prompt_returns_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner, _ = _mock_runner_and_result("hello back")
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        response = await target.send_prompt("hello")
        assert isinstance(response, AgentResponse)
        assert response.text == "hello back"

    @pytest.mark.asyncio
    async def test_first_prompt_sends_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner, _ = _mock_runner_and_result()
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.send_prompt("first")

        call_args = runner.run.call_args
        assert call_args[0][1] == "first"

    @pytest.mark.asyncio
    async def test_second_prompt_sends_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner, _ = _mock_runner_and_result()
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.send_prompt("first")
        await target.send_prompt("second")

        input_data = runner.run.call_args[0][1]
        assert isinstance(input_data, list)
        assert input_data[-1] == {"role": "user", "content": "second"}

    @pytest.mark.asyncio
    async def test_new_returns_fresh_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner, _ = _mock_runner_and_result()
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.send_prompt("first")
        fresh = target.new()

        assert fresh is not target
        assert fresh._history == []

    def test_clone_returns_independent_instance(self) -> None:
        agent = MagicMock()
        target = OpenAIAgentTarget(agent, run_kwargs={"max_turns": 5})
        target._history = [{"role": "user", "content": "old"}]
        cloned = target.new()
        assert cloned is not target
        assert cloned._history == []
        assert cloned._agent is agent
        assert cloned._run_kwargs is not target._run_kwargs
        assert cloned._run_kwargs == {"max_turns": 5}

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
            await target.send_prompt("hi")

    @pytest.mark.asyncio
    async def test_runner_error_is_wrapped_with_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = MagicMock()
        runner.run = AsyncMock(side_effect=RuntimeError("model overloaded"))
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with pytest.raises(RuntimeError, match="OpenAIAgentTarget: Runner.run\\(\\) raised"):
            await target.send_prompt("hi")

    @pytest.mark.asyncio
    async def test_runner_timeout_is_not_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = MagicMock()
        runner.run = AsyncMock(side_effect=asyncio.TimeoutError)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        with pytest.raises(asyncio.TimeoutError):
            await target.send_prompt("hi")

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
        response = await target.send_prompt("find docs")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments == {"query": "tool calls"}

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
        response = await target.send_prompt("run")

        assert response.tool_calls[0].arguments == {"raw": "not-json"}

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

    @pytest.mark.asyncio
    async def test_history_accumulates_across_turns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        async def fake_run(agent, input_data, **kwargs):  # noqa: ANN001, ANN003
            nonlocal call_count
            call_count += 1
            if isinstance(input_data, str):
                history = [
                    {"role": "user", "content": input_data},
                    {"role": "assistant", "content": f"Reply {call_count}"},
                ]
            else:
                history = [*input_data, {"role": "assistant", "content": f"Reply {call_count}"}]
            result = MagicMock()
            result.final_output = f"Reply {call_count}"
            result.to_input_list.return_value = history
            return result

        runner = MagicMock()
        runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        r1 = await target.send_prompt("Hello")
        assert r1.text == "Reply 1"
        r2 = await target.send_prompt("Follow up")
        assert r2.text == "Reply 2"
        assert len(target._history) > 2


class TestOpenAIAgentTargetUsage:
    @pytest.mark.asyncio
    async def test_usage_populated_from_context_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When result.context_wrapper.usage has token counts, SendResult.usage is populated."""
        runner, _ = _mock_runner_with_usage(
            output="hello back",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.send_prompt("hello")

        assert result.text == "hello back"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.usage.calls == 1

    @pytest.mark.asyncio
    async def test_usage_none_when_context_wrapper_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When result has no context_wrapper, SendResult.usage is None (graceful fallback)."""
        runner, _ = _mock_runner_and_result("reply")
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.send_prompt("hi")

        assert result.text == "reply"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_usage_none_when_usage_attr_absent_on_context_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When context_wrapper exists but has no 'usage', SendResult.usage is None."""
        ctx = MagicMock(spec=[])  # spec=[] means no attributes by default

        sdk_result = MagicMock()
        sdk_result.final_output = "hello"
        sdk_result.context_wrapper = ctx
        sdk_result.to_input_list.return_value = []

        runner = MagicMock()
        runner.run = AsyncMock(return_value=sdk_result)
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", runner)

        target = OpenAIAgentTarget(MagicMock())
        result = await target.send_prompt("hi")

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
        result = await target.send_prompt("hi")

        assert result.usage is not None
        assert result.usage.total_tokens == 11  # 8 + 3
