"""`BaseAgent` actually places prompt-cache breakpoints on the wire.

`tests/unit/test_prompt_cache.py` covers the pure placement function; these cover
the wiring — that `_call_chat_completions` marks the messages it sends, that
`_call_responses` carries the top-level switch, that both are router-only, and
that the judge keeps the breakpoint off the instruction it rebuilds every turn.
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.agents.base import BaseAgent
from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.types import Criterion, Message

_ORQ_ROUTER_BASE_URL = 'https://my.orq.ai/v3/router'
_EPHEMERAL = {'type': 'ephemeral'}


class _ConcreteAgent(BaseAgent):
    @property
    def name(self) -> str:
        return 'TestAgent'

    @property
    def system_prompt(self) -> str:
        return 'You are a test agent.'


def _chat_client(base_url: str = _ORQ_ROUTER_BASE_URL) -> MagicMock:
    client = MagicMock()
    client.base_url = base_url
    message = MagicMock(content='reply', tool_calls=None, refusal=None)
    response = MagicMock(choices=[MagicMock(message=message, finish_reason='stop')])
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _responses_client(base_url: str = _ORQ_ROUTER_BASE_URL) -> MagicMock:
    client = MagicMock()
    client.base_url = base_url
    response = MagicMock(output=[], usage=None)
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=response)
    return client


def _sent_messages(client: MagicMock) -> list[dict[str, Any]]:
    return client.chat.completions.create.call_args.kwargs['messages']


@pytest.mark.asyncio
async def test_chat_completions_marks_system_and_last_turn_on_the_router() -> None:
    client = _chat_client()
    agent = _ConcreteAgent(LLMCallConfig(model='gpt-4o', api='chat_completions', client=client))

    await agent._call_llm([Message(role='user', content='hi')])

    messages = _sent_messages(client)
    assert messages[0]['content'] == [
        {'type': 'text', 'text': 'You are a test agent.', 'cache_control': _EPHEMERAL}
    ]
    assert messages[1]['content'] == [{'type': 'text', 'text': 'hi', 'cache_control': _EPHEMERAL}]


@pytest.mark.asyncio
async def test_chat_completions_leaves_a_direct_openai_client_alone() -> None:
    client = _chat_client('https://api.openai.com/v1')
    agent = _ConcreteAgent(LLMCallConfig(model='gpt-4o', api='chat_completions', client=client))

    await agent._call_llm([Message(role='user', content='hi')])

    messages = _sent_messages(client)
    assert messages[0]['content'] == 'You are a test agent.'
    assert messages[1]['content'] == 'hi'


@pytest.mark.asyncio
async def test_volatile_tail_moves_the_breakpoint_off_the_last_message() -> None:
    client = _chat_client()
    agent = _ConcreteAgent(LLMCallConfig(model='gpt-4o', api='chat_completions', client=client))

    await agent._call_llm(
        [Message(role='user', content='persisted'), Message(role='user', content='rebuilt')],
        volatile_tail=1,
    )

    messages = _sent_messages(client)
    assert messages[1]['content'] == [{'type': 'text', 'text': 'persisted', 'cache_control': _EPHEMERAL}]
    assert messages[2]['content'] == 'rebuilt'


@pytest.mark.asyncio
async def test_responses_sends_the_top_level_cache_switch_on_the_router() -> None:
    client = _responses_client()
    agent = _ConcreteAgent(LLMCallConfig(model='gpt-4o', api='responses', client=client))

    await agent._call_llm([Message(role='user', content='hi')])

    assert client.responses.create.call_args.kwargs['extra_body']['cache_control'] == _EPHEMERAL


@pytest.mark.asyncio
async def test_responses_sends_no_cache_switch_to_a_direct_openai_client() -> None:
    client = _responses_client('https://api.openai.com/v1')
    agent = _ConcreteAgent(LLMCallConfig(model='gpt-4o', api='responses', client=client))

    await agent._call_llm([Message(role='user', content='hi')])

    assert 'cache_control' not in client.responses.create.call_args.kwargs.get('extra_body', {})


def _judge(client: MagicMock, api: str = 'chat_completions') -> JudgeAgent:
    return JudgeAgent(
        JudgeAgentConfig(
            model='gpt-4o',
            api=api,
            client=client,
            goal='help the user',
            criteria=[Criterion(description='greets the user', type='must_happen')],
        )
    )


@pytest.mark.asyncio
async def test_judge_puts_the_settled_note_in_the_message_it_sends() -> None:
    """The note only exists to reach the model — assert it on the wire, not by
    calling the renderer."""
    client = _chat_client()
    judge = _judge(client)
    judge.mark_settled({'criteria_0'})

    await judge.evaluate([Message(role='user', content='hello'), Message(role='assistant', content='hi')])

    messages = _sent_messages(client)
    assert 'ALREADY CONFIRMED' in messages[-1]['content']
    assert 'criteria_0' in messages[-1]['content']
    # The static instruction names the marker; no criterion is flagged with it there.
    assert 'ALREADY CONFIRMED: criteria' not in str(messages[0]['content'])


@pytest.mark.asyncio
async def test_judge_keeps_the_breakpoint_off_its_per_turn_instruction() -> None:
    """The instruction is rebuilt every judgement, so marking it would write the
    whole transcript per turn and read none of it back."""
    client = _chat_client()
    judge = _judge(client)

    await judge.evaluate([Message(role='user', content='hello'), Message(role='assistant', content='hi')])

    messages = _sent_messages(client)
    assert isinstance(messages[-1]['content'], str)
    assert messages[1]['content'] == [{'type': 'text', 'text': 'hello', 'cache_control': _EPHEMERAL}]


@pytest.mark.asyncio
async def test_judge_on_responses_warns_that_the_transcript_cannot_cache(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Responses `cache_control` marks the end of the whole input, which the
    per-turn instruction invalidates — a silent full re-encode every judgement."""
    client = _responses_client()
    judge = _judge(client, api='responses')

    with caplog.at_level('WARNING'):
        await judge.evaluate([Message(role='user', content='hello')])
        await judge.evaluate([Message(role='user', content='hello')])

    warnings = [r for r in caplog.records if 'Responses API' in r.getMessage()]
    assert len(warnings) == 1, 'warn once per judge, not once per turn'
    assert "api='chat_completions'" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_judge_on_responses_stays_quiet_off_the_router(caplog: pytest.LogCaptureFixture) -> None:
    client = _responses_client('https://api.openai.com/v1')
    judge = _judge(client, api='responses')

    with caplog.at_level('WARNING'):
        await judge.evaluate([Message(role='user', content='hello')])

    assert not [r for r in caplog.records if 'Responses API' in r.getMessage()]
