"""Tests for from_orq_deployment: token usage surfacing and pipeline/thread tagging."""

# ruff: noqa: S101

import importlib
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.common.thread_context import conversation_thread, evaluatorq_pipeline
from evaluatorq.contracts import AgentResponse, TokenUsage
from evaluatorq.deployment import DeploymentResponse
from evaluatorq.simulation.adapters import from_orq_deployment
from evaluatorq.simulation.types import Message

_deployment_module = importlib.import_module('evaluatorq.deployment')


def _messages() -> list[Message]:
    return [Message(role='user', content='Hello')]


@pytest.mark.asyncio
async def test_from_orq_deployment_returns_agent_response_with_usage() -> None:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_deployment = AsyncMock(return_value=DeploymentResponse(content='hello', raw=None, usage=usage))

    with patch.object(_deployment_module, 'deployment', mock_deployment):
        callback = from_orq_deployment('some-key')
        result = await callback(_messages())

    assert isinstance(result, AgentResponse)
    assert result.text == 'hello'
    assert result.usage is not None
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_from_orq_deployment_tags_pipeline_and_thread() -> None:
    mock_deployment = AsyncMock(return_value=DeploymentResponse(content='hello', raw=None, usage=None))

    with patch.object(_deployment_module, 'deployment', mock_deployment):
        callback = from_orq_deployment('some-key')
        with evaluatorq_pipeline('agent_simulation'), conversation_thread('t-123'):
            await callback(_messages())

    assert mock_deployment.call_args.kwargs['metadata'] == {'evaluatorq_pipeline': 'agent_simulation'}
    assert mock_deployment.call_args.kwargs['thread'] == {'id': 't-123'}


@pytest.mark.asyncio
async def test_from_orq_deployment_no_metadata_without_bound_context() -> None:
    mock_deployment = AsyncMock(return_value=DeploymentResponse(content='hello', raw=None, usage=None))

    with patch.object(_deployment_module, 'deployment', mock_deployment):
        callback = from_orq_deployment('some-key')
        await callback(_messages())

    assert mock_deployment.call_args.kwargs['metadata'] is None
    assert mock_deployment.call_args.kwargs['thread'] is None
