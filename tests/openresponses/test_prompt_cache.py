"""`OrqResponsesTarget` marks the end of the transcript it resends every turn.

This target is both the simulation ``agent:<key>`` target and the execution half
of the default red-team agent backend, so it carries the largest replayed prefix
in either surface. Being stateless is what makes the breakpoint pay: every turn
resends the previous turns verbatim and appends to them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import LLMCallConfig, Message
from evaluatorq.openresponses.input_items import messages_to_responses_input
from evaluatorq.openresponses.target import OrqResponsesTarget

_ROUTER = 'https://my.orq.ai/v3/router'
# Past CACHE_MIN_PROMPT_TOKENS; below it the helper correctly marks nothing.
_LONG = 'the agent must verify the account before quoting a refund. ' * 120


def _resp() -> dict[str, Any]:
    return {
        'id': 'resp_1',
        'object': 'response',
        'model': 'agent/support',
        'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'ok'}]}],
    }


def _client(base_url: str = _ROUTER) -> MagicMock:
    client = MagicMock()
    client.base_url = base_url
    client.responses.create = AsyncMock(return_value=_resp())
    return client


def _transcript() -> list[Message]:
    return [
        Message(role='user', content=f'persisted {_LONG}'),
        Message(role='assistant', content='acknowledged'),
        Message(role='user', content='next turn'),
    ]


async def _sent(client: MagicMock, model: str) -> list[dict[str, Any]]:
    target = OrqResponsesTarget(LLMCallConfig(model=model), client=client)
    await target.respond(_transcript())
    return client.responses.create.call_args.kwargs['input']


@pytest.mark.asyncio
async def test_agent_target_marks_the_end_of_the_transcript() -> None:
    """volatile_items=0: the caller owns the transcript and this target only
    appends, so the whole input persists into the next turn."""
    sent = await _sent(_client(), 'agent/support-bot')

    assert sent[-1]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
    # Only the prefix end; earlier turns stay plain (max 4 breakpoints).
    assert 'cache_control' not in str(sent[:-1])


@pytest.mark.asyncio
async def test_agent_target_leaves_a_direct_openai_client_alone() -> None:
    """Asserted as an equality against the unmarked render: a plain user turn is
    a bare *string*, so an `'cache_control' not in part` loop over it would
    iterate characters and pass however the input was marked."""
    client = _client('https://api.openai.com/v1')
    sent = await _sent(client, 'agent/support-bot')

    assert sent == messages_to_responses_input(_transcript())


@pytest.mark.asyncio
async def test_agent_target_leaves_a_routed_non_anthropic_model_alone() -> None:
    sent = await _sent(_client(), 'openai/gpt-4o')

    assert sent == messages_to_responses_input(_transcript())


@pytest.mark.asyncio
async def test_agent_target_does_not_mark_a_short_exchange() -> None:
    """Below the provider minimum a breakpoint is a 1.25x write nothing reads."""
    client = _client()
    target = OrqResponsesTarget(LLMCallConfig(model='agent/support-bot'), client=client)
    messages = [Message(role='user', content='hi')]

    await target.respond(messages)

    assert client.responses.create.call_args.kwargs['input'] == messages_to_responses_input(messages)
