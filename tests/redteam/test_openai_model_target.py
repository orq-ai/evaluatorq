# tests/redteam/test_openai_model_target.py
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from evaluatorq.redteam.backends.openai import OpenAIModelTarget
from evaluatorq.redteam.contracts import AgentContext, TargetKind

pytest.importorskip("openai")


def _make_completion_response(
    content: str = "reply",
    tool_calls: list[Any] | None = None,
    model: str = "gpt-4o",
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.id = "chatcmpl-test"
    return response


def _make_tool_call(id: str = "tc-1", name: str = "search", arguments: str = '{"q":"hi"}') -> MagicMock:
    func = MagicMock()
    func.name = name
    func.arguments = arguments

    tc = MagicMock()
    tc.id = id
    tc.function = func
    tc.type = "function"
    return tc


def test_optional_client_auto_creates():
    with patch('evaluatorq.redteam.backends.openai.create_async_llm_client') as mock_create:
        mock_client = MagicMock()
        mock_create.return_value = mock_client
        target = OpenAIModelTarget(model='gpt-4o')
        mock_create.assert_called_once()
        assert target.client is mock_client


def test_explicit_client_skips_auto_create():
    with patch('evaluatorq.redteam.backends.openai.create_async_llm_client') as mock_create:
        client = MagicMock()
        target = OpenAIModelTarget(model='gpt-4o', client=client)
        mock_create.assert_not_called()
        assert target.client is client


def test_model_param_name():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-4o', client=client)
    assert target.model == 'gpt-4o'


def test_system_prompt_default():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-4o', client=client)
    assert target.system_prompt == 'You are a helpful assistant.'


def test_new_preserves_fields():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-4o', system_prompt='Be terse.', client=client)
    fresh = target.new()
    assert fresh.model == 'gpt-4o'
    assert fresh.system_prompt == 'Be terse.'
    assert fresh.client is client


def test_target_kind_is_openai():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-4o', client=client)
    assert target.target_kind == TargetKind.OPENAI


def test_name_returns_model():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-5-mini', client=client)
    assert target.name == 'gpt-5-mini'


@pytest.mark.asyncio
async def test_get_agent_context_returns_agent_context():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-4o', client=client)
    ctx = await target.get_agent_context()
    assert isinstance(ctx, AgentContext)
    assert ctx.key == 'gpt-4o'


# ===========================================================================
# Multi-turn history
# ===========================================================================


class TestOpenAIModelTargetHistory:
    def _make_target(self, content: str = "reply") -> tuple[OpenAIModelTarget, MagicMock]:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response(content=content)
        )
        target = OpenAIModelTarget(model="gpt-4o", client=client)
        return target, client

    def test_history_starts_empty(self) -> None:
        target, _ = self._make_target()
        assert target._history == []

    @pytest.mark.asyncio
    async def test_first_prompt_appends_user_and_assistant(self) -> None:
        target, _ = self._make_target(content="hello back")
        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            await target.send_prompt("hello")
        assert len(target._history) == 2
        assert target._history[0] == {"role": "user", "content": "hello"}
        assert target._history[1] == {"role": "assistant", "content": "hello back"}

    @pytest.mark.asyncio
    async def test_second_prompt_includes_prior_history_in_api_call(self) -> None:
        target, client = self._make_target()
        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            await target.send_prompt("turn one")
            await target.send_prompt("turn two")

        calls = client.chat.completions.create.call_args_list
        assert len(calls) == 2

        # Second call must include system + prior user + prior assistant + new user
        second_messages = calls[1].kwargs["messages"]
        roles = [m["role"] for m in second_messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert second_messages[-1]["content"] == "turn two"

    @pytest.mark.asyncio
    async def test_history_grows_across_turns(self) -> None:
        target, _ = self._make_target()
        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            await target.send_prompt("a")
            await target.send_prompt("b")
            await target.send_prompt("c")
        # 3 turns × 2 messages each
        assert len(target._history) == 6
        roles = [m["role"] for m in target._history]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_tool_calls_recorded_in_history(self) -> None:
        tc = _make_tool_call(id="tc-1", name="search", arguments='{"q":"test"}')
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response(content="", tool_calls=[tc])
        )
        target = OpenAIModelTarget(model="gpt-4o", client=client)

        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            await target.send_prompt("search for something")

        assistant_msg = target._history[1]
        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["id"] == "tc-1"  # pyright: ignore[reportIndexIssue]
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "search"  # pyright: ignore[reportIndexIssue]
        assert assistant_msg["tool_calls"][0]["type"] == "function"  # pyright: ignore[reportIndexIssue]

    def test_new_returns_fresh_empty_history(self) -> None:
        client = MagicMock()
        target = OpenAIModelTarget(model="gpt-4o", client=client)
        # Directly inject history to simulate prior turns
        target._history = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old reply"}]
        fresh = target.new()
        assert fresh._history == []

    @pytest.mark.asyncio
    async def test_first_call_sends_only_system_and_user(self) -> None:
        target, client = self._make_target()
        with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
            await target.send_prompt("first message")

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "first message"
        assert len(messages) == 2
