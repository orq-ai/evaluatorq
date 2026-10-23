"""Every place a recorded transcript leaves for a provider drops unpaired tool records.

The in-loop judge is the one reader that must still SEE an incomplete call, so it gets the
call as text; targets get nothing for it.
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.common.target_call import call_target_with_retry
from evaluatorq.contracts import AgentResponse, FunctionCall, LLMCallConfig, StrategyToolCall
from evaluatorq.simulation.agents.base import BaseAgent
from evaluatorq.simulation.agents.judge import JudgeAgent
from evaluatorq.simulation.contracts import Criterion, Message
from evaluatorq.simulation.runner.simulation import _invert_roles_for_simulator

_MARKER = '[tool_call: refund({}) -> no result recorded]'


def _transcript() -> list[Message]:
    return [
        Message(role='user', content='refund me'),
        Message(
            role='assistant',
            content='Checking.',
            tool_calls=[
                StrategyToolCall(id='unpaired', function=FunctionCall(name='refund', arguments='{}'), incomplete=True),
                StrategyToolCall(id='paired', function=FunctionCall(name='lookup', arguments='{}')),
            ],
        ),
        Message(role='tool', tool_call_id='paired', name='lookup', content='ok'),
        Message(role='tool', tool_call_id='stray', content='orphan'),
    ]


class _Agent(BaseAgent):
    @property
    def name(self) -> str:
        return 'TestAgent'

    @property
    def system_prompt(self) -> str:
        return 'You are a test agent.'


def _chat_client() -> MagicMock:
    client = MagicMock()
    client.base_url = 'https://api.openai.com/v1'
    message = MagicMock(content='reply', tool_calls=None, refusal=None)
    response = MagicMock(choices=[MagicMock(message=message, finish_reason='stop')])
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _responses_client() -> MagicMock:
    client = MagicMock()
    client.base_url = 'https://api.openai.com/v1'
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=MagicMock(output=[], usage=None))
    return client


def _call_ids(payload: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    calls = {
        c['id'] for m in payload if m.get('role') == 'assistant' for c in m.get('tool_calls') or []
    } | {i['call_id'] for i in payload if i.get('type') == 'function_call'}
    results = {m['tool_call_id'] for m in payload if m.get('role') == 'tool'} | {
        i['call_id'] for i in payload if i.get('type') == 'function_call_output'
    }
    return calls, results


@pytest.mark.asyncio
@pytest.mark.parametrize('api', ['chat_completions', 'responses'])
async def test_agent_sends_no_unpaired_id_and_shows_the_attempt_as_text(api):
    client = _chat_client() if api == 'chat_completions' else _responses_client()
    agent = _Agent(LLMCallConfig(model='openai/gpt-4o', api=api, client=client))

    await agent._call_llm(_transcript())

    if api == 'chat_completions':
        payload = client.chat.completions.create.call_args.kwargs['messages']
    else:
        payload = client.responses.create.call_args.kwargs['input']
    assert _call_ids(payload) == ({'paired'}, {'paired'})
    assert _MARKER in str(payload)


@pytest.mark.asyncio
async def test_judge_evaluate_sees_the_incomplete_call():
    client = _responses_client()
    judge = JudgeAgent(
        LLMCallConfig(
            model='openai/gpt-4o',
            client=client,
        ),
        goal='get a refund',
        criteria=[Criterion(description='refunds', type='must_happen')],
    )

    await judge.evaluate(_transcript())

    payload = client.responses.create.call_args.kwargs['input']
    assert _MARKER in str(payload)
    assert _call_ids(payload) == ({'paired'}, {'paired'})


@pytest.mark.asyncio
async def test_target_choke_point_sends_a_filtered_copy_and_leaves_the_evidence_alone():
    seen: list[list[Message]] = []

    class _Target:
        async def respond(self, messages: list[Message]) -> AgentResponse:
            seen.append(messages)
            return AgentResponse()

    transcript = _transcript()
    snapshot = [m.model_copy(deep=True) for m in transcript]

    await call_target_with_retry(_Target(), transcript, target_agent_timeout_ms=1000, max_target_retries=0)

    sent = [m.model_dump() for m in seen[0]]
    assert [m['role'] for m in sent] == ['user', 'assistant', 'tool']
    assert [tc['id'] for tc in sent[1]['tool_calls']] == ['paired']
    assert transcript == snapshot


def test_user_simulator_view_carries_no_tool_records():
    inverted = _invert_roles_for_simulator(_transcript())

    assert all(m.role != 'tool' and not m.tool_calls for m in inverted)
