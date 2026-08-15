"""Red-team target backends tag every target LLM call with the active run id.

Covers the two distinct mechanisms red-team backends use to merge
``evaluatorq_run_id`` correlation into the outbound request (see
``evaluatorq.common.thread_context`` / ``evaluatorq.common.llm_call``):

- ``OpenAIModelTarget.respond`` (``redteam/backends/openai.py``) merges
  ``pipeline_metadata()`` as a native ``metadata=`` kwarg on every endpoint.
- ``ORQAgentTarget.respond`` (``redteam/backends/orq.py``) merges
  ``pipeline_metadata_param()`` (an ``{'metadata': {...}}``-shaped dict) directly
  into the ORQ SDK call kwargs at both call sites (the initial turn and the
  pending-tool-call continuation loop) — unconditionally, since the ORQ agents
  endpoint always routes through Orq. Unlike the OpenAI chat-completions shape,
  the ORQ SDK's ``agents.responses.create`` takes ``metadata`` as a top-level
  kwarg, not nested under ``extra_body``.

Modeled on ``tests/simulation/test_pipeline_metadata.py`` (same claim, sim
side) and ``tests/redteam/test_backends.py`` (existing redteam backend fakes).
"""

from __future__ import annotations

# ruff: noqa: S101
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.common.thread_context import evaluatorq_pipeline, evaluatorq_run_id
from evaluatorq.contracts import Message
from evaluatorq.redteam.backends.openai import OpenAIModelTarget
from evaluatorq.redteam.backends.orq import ORQAgentTarget

_ORQ_ROUTER_BASE_URL = 'https://my.orq.ai/v3/router'


def _openai_response(content: str = 'hi there') -> MagicMock:
    msg = MagicMock(content=content, tool_calls=None)
    choice = MagicMock(message=msg, finish_reason='stop')
    resp = MagicMock(choices=[choice], id='resp_1', model='gpt-4o')
    resp.usage = None
    return resp


@pytest.mark.asyncio
class TestOpenAIModelTargetRunMetadata:
    """``metadata=`` kwarg path (native OpenAI chat-completions field)."""

    async def test_sends_run_metadata_when_bound_and_orq_routed(self) -> None:
        client = MagicMock()
        client.base_url = _ORQ_ROUTER_BASE_URL
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        target = OpenAIModelTarget(model='gpt-4o-mini', client=client)

        with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('rt-run-1'):
            await target.respond([Message(role='user', content='hello')])

        _, kwargs = client.chat.completions.create.call_args
        assert kwargs.get('metadata') == {
            'evaluatorq_pipeline': 'red_teaming',
            'evaluatorq_run_id': 'rt-run-1',
        }

    async def test_sends_metadata_when_client_not_orq_routed(self) -> None:
        """Native OpenAI metadata is endpoint-neutral."""
        client = MagicMock()
        client.base_url = 'https://api.openai.com/v1'
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        target = OpenAIModelTarget(model='gpt-4o-mini', client=client)

        with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('rt-run-1'):
            await target.respond([Message(role='user', content='hello')])

        _, kwargs = client.chat.completions.create.call_args
        assert kwargs['metadata'] == {
            'evaluatorq_pipeline': 'red_teaming',
            'evaluatorq_run_id': 'rt-run-1',
        }

    async def test_no_metadata_without_bound_run_id(self) -> None:
        client = MagicMock()
        client.base_url = _ORQ_ROUTER_BASE_URL
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        target = OpenAIModelTarget(model='gpt-4o-mini', client=client)

        await target.respond([Message(role='user', content='hello')])

        _, kwargs = client.chat.completions.create.call_args
        assert 'metadata' not in kwargs


def _orq_response(*, pending_tool_calls: list[Any] | None = None, text: str = 'ok') -> MagicMock:
    resp = MagicMock()
    resp.task_id = 'task_1'
    resp.pending_tool_calls = pending_tool_calls or []
    resp.usage = None
    resp.model = None
    resp.telemetry = None
    part = MagicMock(kind='text', text=text)
    item = MagicMock(parts=[part])
    resp.output = [item]
    return resp


@pytest.mark.asyncio
class TestORQAgentTargetRunMetadata:
    """``extra_body`` path (``pipeline_metadata_param``), both call sites."""

    async def test_initial_call_carries_run_metadata_in_extra_body(self) -> None:
        orq_client = MagicMock()
        orq_client.agents.responses.create = MagicMock(return_value=_orq_response())
        target = ORQAgentTarget(agent_key='test-agent', orq_client=orq_client)

        async def fake_to_thread(fn, **kwargs):
            return fn(**kwargs)

        with (
            patch('evaluatorq.redteam.backends.orq.asyncio.to_thread', side_effect=fake_to_thread),
            evaluatorq_pipeline('red_teaming'),
            evaluatorq_run_id('rt-run-2'),
        ):
            await target.respond([Message(role='user', content='hello')])

        assert orq_client.agents.responses.create.call_count == 1
        _, kwargs = orq_client.agents.responses.create.call_args
        assert kwargs['retries'] is None
        assert kwargs['metadata'] == {
            'evaluatorq_pipeline': 'red_teaming',
            'evaluatorq_run_id': 'rt-run-2',
        }

    async def test_continuation_call_carries_run_metadata_in_extra_body(self) -> None:
        """The tool-call continuation loop (second call site, orq.py:298) must
        independently carry the same run metadata as the initial call."""
        orq_client = MagicMock()
        pending_call = MagicMock(id='call_1', name='search', arguments={'query': 'x'})
        first_resp = _orq_response(pending_tool_calls=[pending_call])
        second_resp = _orq_response(text='done')
        orq_client.agents.responses.create = MagicMock(side_effect=[first_resp, second_resp])
        target = ORQAgentTarget(agent_key='test-agent', orq_client=orq_client)

        async def fake_to_thread(fn, **kwargs):
            return fn(**kwargs)

        with (
            patch('evaluatorq.redteam.backends.orq.asyncio.to_thread', side_effect=fake_to_thread),
            evaluatorq_pipeline('red_teaming'),
            evaluatorq_run_id('rt-run-3'),
        ):
            await target.respond([Message(role='user', content='hello')])

        assert orq_client.agents.responses.create.call_count == 2
        _, continuation_kwargs = orq_client.agents.responses.create.call_args_list[1]
        assert continuation_kwargs['retries'] is None
        assert continuation_kwargs['metadata'] == {
            'evaluatorq_pipeline': 'red_teaming',
            'evaluatorq_run_id': 'rt-run-3',
        }

    async def test_no_metadata_without_bound_run_id(self) -> None:
        orq_client = MagicMock()
        orq_client.agents.responses.create = MagicMock(return_value=_orq_response())
        target = ORQAgentTarget(agent_key='test-agent', orq_client=orq_client)

        async def fake_to_thread(fn, **kwargs):
            return fn(**kwargs)

        with patch('evaluatorq.redteam.backends.orq.asyncio.to_thread', side_effect=fake_to_thread):
            await target.respond([Message(role='user', content='hello')])

        _, kwargs = orq_client.agents.responses.create.call_args
        assert 'metadata' not in kwargs
