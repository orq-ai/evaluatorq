"""Unit tests for OpenAI Agents SDK red teaming target."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("agents")

from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget  # noqa: E402


def _mock_runner_and_result(output: str = "response") -> tuple[MagicMock, MagicMock]:
    result = MagicMock()
    result.final_output = output
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
        assert response == "hello back"

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
        assert r1 == "Reply 1"
        r2 = await target.send_prompt("Follow up")
        assert r2 == "Reply 2"
        assert len(target._history) > 2
