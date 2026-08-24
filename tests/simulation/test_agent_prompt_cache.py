"""`BaseAgent` actually places prompt-cache breakpoints on the wire.

`tests/unit/test_prompt_cache.py` covers the pure placement functions; these
cover the wiring — that `_call_chat_completions` and `_call_responses` both mark
the end of the persisted prefix, that both are gated on router + model, and that
the judge keeps the breakpoint off the instruction it rebuilds every turn.

Every transcript here is padded past `CACHE_MIN_PROMPT_TOKENS`: below that the
helpers correctly place nothing, so an unpadded fixture would make every
assertion here pass for the wrong reason.
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.openresponses.input_items import messages_to_responses_input
from evaluatorq.simulation.agents.base import BaseAgent
from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.types import Criterion, Message

_ORQ_ROUTER_BASE_URL = 'https://my.orq.ai/v3/router'
_EPHEMERAL = {'type': 'ephemeral'}
_CACHED_MODEL = 'anthropic/claude-sonnet-4-6'
_UNCACHED_MODEL = 'openai/gpt-4o'
# Comfortably past CACHE_MIN_PROMPT_TOKENS * 4 chars, so the size guard never
# silently turns one of these tests into a no-op.
_LONG = 'the support agent must verify the account before quoting a refund. ' * 120


class _ConcreteAgent(BaseAgent):
    @property
    def name(self) -> str:
        return 'TestAgent'

    @property
    def system_prompt(self) -> str:
        return f'You are a test agent. {_LONG}'


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


def _agent(
    client: MagicMock,
    *,
    api: Literal['chat_completions', 'responses'] = 'chat_completions',
    model: str = _CACHED_MODEL,
) -> _ConcreteAgent:
    return _ConcreteAgent(LLMCallConfig(model=model, api=api, client=client))


def _transcript() -> list[Message]:
    """A transcript that renders to parts-shaped Responses items, not bare strings.

    An assistant turn is the only role `messages_to_responses_input` renders as a
    content-part list. Without one, every `'cache_control' not in part` assertion
    below would iterate the characters of a string and pass unconditionally.
    """
    return [
        Message(role='user', content=f'persisted {_LONG}'),
        Message(role='assistant', content='acknowledged'),
        Message(role='user', content='rebuilt'),
    ]


@pytest.mark.asyncio
async def test_chat_completions_marks_system_and_last_turn_on_the_router() -> None:
    client = _chat_client()

    await _agent(client)._call_llm([Message(role='user', content=f'hi {_LONG}')])

    messages = _sent_messages(client)
    assert messages[0]['content'] == [
        {'type': 'text', 'text': f'You are a test agent. {_LONG}', 'cache_control': _EPHEMERAL}
    ]
    assert messages[1]['content'] == [{'type': 'text', 'text': f'hi {_LONG}', 'cache_control': _EPHEMERAL}]


@pytest.mark.asyncio
async def test_chat_completions_leaves_a_direct_openai_client_alone() -> None:
    client = _chat_client('https://api.openai.com/v1')

    await _agent(client)._call_llm([Message(role='user', content=f'hi {_LONG}')])

    messages = _sent_messages(client)
    assert messages[0]['content'] == f'You are a test agent. {_LONG}'
    assert messages[1]['content'] == f'hi {_LONG}'


@pytest.mark.asyncio
async def test_chat_completions_leaves_a_non_anthropic_routed_model_alone() -> None:
    """Routed, but OpenAI caches automatically — a marker there is shape churn."""
    client = _chat_client()

    await _agent(client, model=_UNCACHED_MODEL)._call_llm([Message(role='user', content=f'hi {_LONG}')])

    messages = _sent_messages(client)
    assert messages[0]['content'] == f'You are a test agent. {_LONG}'
    assert messages[1]['content'] == f'hi {_LONG}'


@pytest.mark.asyncio
async def test_a_short_prompt_is_not_marked_at_all() -> None:
    """Below the provider minimum a breakpoint is a 1.25x write nothing reads."""
    client = _chat_client()

    class _ShortAgent(_ConcreteAgent):
        @property
        def system_prompt(self) -> str:
            return 'short'

    await _ShortAgent(LLMCallConfig(model=_CACHED_MODEL, api='chat_completions', client=client))._call_llm(
        [Message(role='user', content='hi')]
    )

    messages = _sent_messages(client)
    assert messages[0]['content'] == 'short'
    assert messages[1]['content'] == 'hi'


@pytest.mark.asyncio
async def test_volatile_tail_moves_the_breakpoint_off_the_last_message() -> None:
    client = _chat_client()

    await _agent(client)._call_llm(
        [Message(role='user', content=f'persisted {_LONG}'), Message(role='user', content='rebuilt')],
        volatile_tail=1,
    )

    messages = _sent_messages(client)
    assert messages[1]['content'] == [
        {'type': 'text', 'text': f'persisted {_LONG}', 'cache_control': _EPHEMERAL}
    ]
    assert messages[2]['content'] == 'rebuilt'


@pytest.mark.asyncio
async def test_responses_marks_a_positioned_breakpoint_on_the_router() -> None:
    """Per-item, not the top-level switch: the switch marks the end of the whole
    input, so a caller that rebuilds its trailing item writes every turn and reads
    none."""
    client = _responses_client()

    await _agent(client, api='responses')._call_llm(_transcript(), volatile_tail=1)

    sent = client.responses.create.call_args.kwargs
    # The assistant turn is the prefix end after excluding the rebuilt tail.
    assert sent['input'][1]['content'][-1]['cache_control'] == _EPHEMERAL
    assert 'cache_control' not in sent['input'][2]['content']
    # The top-level switch would add a second, unreadable breakpoint at the end.
    assert 'cache_control' not in sent.get('extra_body', {})


@pytest.mark.asyncio
async def test_responses_volatile_tail_counts_messages_not_items() -> None:
    """One tool-calling assistant `Message` renders to several `input` items.

    Passing the message count straight through would land the breakpoint inside
    the region the caller rebuilds — the failure the whole keyword exists to stop.
    """
    from evaluatorq.contracts import FunctionCall, StrategyToolCall

    messages = [
        Message(role='user', content=f'persisted {_LONG}'),
        Message(
            role='assistant',
            content='calling a tool',
            tool_calls=[StrategyToolCall(id='call_1', function=FunctionCall(name='lookup', arguments='{}'))],
        ),
        Message(role='user', content='rebuilt'),
    ]
    client = _responses_client()

    # The last two messages render to three items (assistant text + function_call
    # + the rebuilt user turn), so the breakpoint must land on the first item.
    await _agent(client, api='responses')._call_llm(messages, volatile_tail=2)

    sent = client.responses.create.call_args.kwargs
    assert len(sent['input']) == 4
    assert sent['input'][0]['content'][-1]['cache_control'] == _EPHEMERAL
    assert not any('cache_control' in str(item) for item in sent['input'][1:])


@pytest.mark.asyncio
async def test_responses_sends_nothing_extra_to_a_direct_openai_client() -> None:
    """Asserted as an equality against the unmarked render, not by absence.

    A plain user turn renders as a bare *string*, so `'cache_control' not in part`
    would iterate characters and pass however the input was marked.
    """
    client = _responses_client('https://api.openai.com/v1')
    messages = _transcript()

    await _agent(client, api='responses')._call_llm(messages, volatile_tail=1)

    sent = client.responses.create.call_args.kwargs
    assert sent['input'] == messages_to_responses_input(messages)
    assert 'cache_control' not in sent.get('extra_body', {})


def _judge(client: MagicMock, api: str = 'chat_completions') -> JudgeAgent:
    return JudgeAgent(
        JudgeAgentConfig(
            model=_CACHED_MODEL,
            api=api,
            client=client,
            goal=f'help the user. {_LONG}',
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
    # Where the note must NOT appear is the system prompt, and the assertion that
    # actually pins that is byte-stability across turns — the static instruction
    # names the marker and the criteria listing names the id, so neither literal
    # can distinguish "flagged here" from "mentioned here".


@pytest.mark.asyncio
async def test_judge_system_prompt_is_byte_stable_across_a_settled_criterion() -> None:
    """The property the cache actually depends on: nothing at token position 0
    changes between turns, whatever the judge learns."""
    client = _chat_client()
    judge = _judge(client)

    await judge.evaluate([Message(role='user', content='hello'), Message(role='assistant', content='hi')])
    before = _sent_messages(client)[0]['content']
    judge.mark_settled({'criteria_0'})
    await judge.evaluate(
        [
            Message(role='user', content='hello'),
            Message(role='assistant', content='hi'),
            Message(role='user', content='and again'),
        ]
    )

    assert _sent_messages(client)[0]['content'] == before


@pytest.mark.asyncio
async def test_judge_keeps_the_breakpoint_off_its_per_turn_instruction() -> None:
    """The instruction is rebuilt every judgement, so marking it would write the
    whole transcript per turn and read none of it back."""
    client = _chat_client()
    judge = _judge(client)

    await judge.evaluate(
        [Message(role='user', content=f'hello {_LONG}'), Message(role='assistant', content='hi')]
    )

    messages = _sent_messages(client)
    assert isinstance(messages[-1]['content'], str)
    assert messages[1]['content'] == [
        {'type': 'text', 'text': f'hello {_LONG}', 'cache_control': _EPHEMERAL}
    ]


@pytest.mark.asyncio
async def test_judge_on_responses_also_keeps_the_breakpoint_off_the_instruction() -> None:
    """Same guarantee as the chat path: the router honours a per-item
    `cache_control`, so the transcript caches and the rebuilt instruction stays
    outside the marked prefix."""
    client = _responses_client()
    judge = _judge(client, api='responses')

    await judge.evaluate(
        [Message(role='user', content=f'hello {_LONG}'), Message(role='assistant', content='hi')]
    )

    items = client.responses.create.call_args.kwargs['input']
    assert items[-2]['content'][-1]['cache_control'] == _EPHEMERAL
    assert 'cache_control' not in str(items[-1])


@pytest.mark.asyncio
async def test_judge_on_responses_marks_nothing_off_the_router() -> None:
    client = _responses_client('https://api.openai.com/v1')
    judge = _judge(client, api='responses')

    await judge.evaluate(
        [Message(role='user', content=f'hello {_LONG}'), Message(role='assistant', content='hi')]
    )

    sent = client.responses.create.call_args.kwargs
    assert 'cache_control' not in str(sent['input'])
