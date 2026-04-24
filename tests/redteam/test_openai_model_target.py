# tests/redteam/test_openai_model_target.py
import pytest
from unittest.mock import MagicMock, patch
from evaluatorq.redteam.backends.openai import OpenAIModelTarget
from evaluatorq.redteam.contracts import AgentContext, TargetKind


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


def test_clone_preserves_fields():
    client = MagicMock()
    target = OpenAIModelTarget(model='gpt-4o', system_prompt='Be terse.', client=client)
    clone = target.clone()
    assert clone.model == 'gpt-4o'
    assert clone.system_prompt == 'Be terse.'
    assert clone.client is client


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
