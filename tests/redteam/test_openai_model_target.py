# tests/redteam/test_openai_model_target.py
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from evaluatorq.contracts import Message
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


@pytest.mark.asyncio
async def test_respond_sends_only_system_and_user() -> None:
    """respond() prepends exactly one system message and the given user message."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_completion_response(content="reply")
    )
    target = OpenAIModelTarget(model="gpt-4o", client=client)
    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond([Message(role="user", content="first message")])

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "first message"
    assert len(messages) == 2
