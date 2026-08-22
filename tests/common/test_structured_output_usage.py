from __future__ import annotations

# ruff: noqa: S101
"""RES-1295: ``generate_structured`` reports what the whole ladder cost.

Fixtures are built from the OpenAI SDK's own response models (``ChatCompletion``,
``ParsedChatCompletion``, ``Response``) rather than from hand-written dicts or
MagicMocks, so a usage-shape move in the SDK fails these tests instead of
confirming a guess about where the numbers live.
"""

import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.parsed_chat_completion import (
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
    ParsedChoice,
)
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from evaluatorq.common.model_catalogue import ModelInfo, register_model
from evaluatorq.common.structured_output import StructuredResult, generate_structured, sum_structured_usage

MODEL = 'test/usage-model'


class SampleResponse(BaseModel):
    value: str


def _usage(prompt: int, completion: int) -> CompletionUsage:
    return CompletionUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


def _parsed_completion(
    parsed: SampleResponse | None,
    *,
    prompt: int,
    completion: int,
) -> ParsedChatCompletion[SampleResponse]:
    """A ``.parse()`` response — rung 1/3's shape, straight from the SDK model."""
    return ParsedChatCompletion[SampleResponse](
        id='cmpl-parse',
        created=0,
        model=MODEL,
        object='chat.completion',
        choices=[
            ParsedChoice[SampleResponse](
                index=0,
                finish_reason='stop',
                message=ParsedChatCompletionMessage[SampleResponse](role='assistant', content=None, parsed=parsed),
            )
        ],
        usage=_usage(prompt, completion),
    )


def _text_completion(content: str, *, prompt: int, completion: int, usage: bool = True) -> ChatCompletion:
    """A ``.create()`` response — rung 2/4's shape."""
    return ChatCompletion(
        id='cmpl-text',
        created=0,
        model=MODEL,
        object='chat.completion',
        choices=[
            Choice(
                index=0,
                finish_reason='stop',
                message=ChatCompletionMessage(role='assistant', content=content),
            )
        ],
        usage=_usage(prompt, completion) if usage else None,
    )


def _responses_response(text: str, *, input_tokens: int, output_tokens: int) -> Response:
    return Response(
        id='resp-1',
        created_at=0,
        model=MODEL,
        object='response',
        parallel_tool_calls=False,
        tool_choice='auto',
        tools=[],
        output=[
            ResponseOutputMessage(
                id='msg-1',
                role='assistant',
                status='completed',
                type='message',
                content=[ResponseOutputText(type='output_text', text=text, annotations=[])],
            )
        ],
        status='completed',
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )


async def _generate(client: Any, **kwargs: Any) -> Any:
    return await generate_structured(
        client,
        model=MODEL,
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='usage-test',
        **kwargs,
    )


@pytest.mark.asyncio
async def test_usage_is_summed_across_every_rung_not_just_the_answering_one() -> None:
    """Rungs 1-3 burn tokens before rung 4 answers; all four are billed.

    Reporting only the rung that answered would understate the call by three
    quarters here — the exact failure RES-1295 describes.
    """
    client = MagicMock()
    # Rung 1 answers with no `parsed` (ladder continues), rung 3 with no tool_calls.
    client.chat.completions.parse = AsyncMock(
        side_effect=[
            _parsed_completion(None, prompt=10, completion=1),
            _parsed_completion(None, prompt=30, completion=3),
        ]
    )
    # Rung 2's content does not validate; rung 4's does.
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _text_completion('not json at all', prompt=20, completion=2),
            _text_completion('{"value": "ok"}', prompt=40, completion=4),
        ]
    )

    result = await _generate(client)

    assert result.parsed is not None
    assert result.parsed.value == 'ok'
    assert result.usage is not None
    assert result.usage.calls == 4
    assert result.usage.input_tokens == 10 + 20 + 30 + 40
    assert result.usage.output_tokens == 1 + 2 + 3 + 4
    assert result.usage.total_tokens == 110


@pytest.mark.asyncio
async def test_usage_covers_the_rungs_burned_when_nothing_validates() -> None:
    """A call that ends with ``parsed is None`` still paid for every rung it ran."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=[
            _parsed_completion(None, prompt=10, completion=1),
            _parsed_completion(None, prompt=30, completion=3),
        ]
    )
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _text_completion('nope', prompt=20, completion=2),
            _text_completion('still nope', prompt=40, completion=4),
        ]
    )

    result = await _generate(client)

    assert result.parsed is None
    assert result.usage is not None
    assert result.usage.calls == 4
    assert result.usage.total_tokens == 110


@pytest.mark.asyncio
async def test_responses_leg_reports_its_own_usage() -> None:
    """The Responses leg's usage comes back on the success path."""
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_responses_response('{"value": "ok"}', input_tokens=11, output_tokens=7))

    result = await _generate(client, api='responses')

    assert result.parsed is not None
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.calls == 1
    client.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_responses_leg_usage_survives_the_fall_through_to_chat() -> None:
    """A Responses leg that answered unusably still billed — its tokens stay in the total."""
    client = MagicMock()
    # No output text -> the leg degrades to the chat ladder, but the call was paid for.
    client.responses.create = AsyncMock(return_value=_responses_response('', input_tokens=11, output_tokens=7))
    client.chat.completions.parse = AsyncMock(
        return_value=_parsed_completion(SampleResponse(value='ok'), prompt=5, completion=2)
    )

    result = await _generate(client, api='responses')

    assert result.parsed is not None
    assert result.usage is not None
    assert result.usage.calls == 2
    assert result.usage.input_tokens == 11 + 5
    assert result.usage.output_tokens == 7 + 2


@pytest.mark.asyncio
async def test_unreadable_usage_counts_as_one_unpriced_call_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rung with no readable usage is unknown, never zero.

    ``calls`` still counts it, ``priced_calls`` does not, and the cause is
    logged — silently dropping it would report a four-rung call as a one-rung
    one, and recording it as $0 would assert a price nobody quoted.
    """
    register_model(
        MODEL,
        ModelInfo(input_cost_per_1k=1.0, output_cost_per_1k=2.0, provider='test', supports_responses=False),
    )
    client = MagicMock()
    # Rung 1 answers with a usage block the extractor cannot read (absent).
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(None, prompt=0, completion=0))
    client.chat.completions.create = AsyncMock(
        return_value=_text_completion('{"value": "ok"}', prompt=1000, completion=1000)
    )

    with caplog.at_level(logging.WARNING):
        result = await _generate(client)

    assert 'no readable usage block' in caplog.text
    assert result.usage is not None
    assert result.usage.calls == 2
    # Only the second rung could be priced, so the cost is a lower bound and says so.
    assert result.usage.priced_calls == 1
    assert result.usage.total_cost == pytest.approx(1.0 + 2.0)
    assert result.usage.cost_is_partial is True


@pytest.mark.asyncio
async def test_a_response_without_a_usage_block_still_counts_as_a_call() -> None:
    """A `usage=None` payload on an otherwise fine response is still one billed call."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(None, prompt=3, completion=1))
    client.chat.completions.create = AsyncMock(
        return_value=_text_completion('{"value": "ok"}', prompt=0, completion=0, usage=False)
    )

    result = await _generate(client)

    assert result.parsed is not None
    assert result.usage is not None
    assert result.usage.calls == 2
    assert result.usage.total_tokens == 4


def test_sum_structured_usage_returns_none_when_nothing_was_billed() -> None:
    assert sum_structured_usage([None, None]) is None


# --- call sites propagate what they receive ------------------------------------


@pytest.mark.asyncio
async def test_persona_generator_accumulates_the_usage_it_is_handed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PersonaGenerator.get_usage()` reflects every call it made, not the last one."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.generators import persona_generator as mod

    class _Parsed:
        personas: list[Any] = []

    async def fake(*_args: Any, **_kwargs: Any) -> StructuredResult[Any]:
        return StructuredResult(
            cast('Any', _Parsed()), '', TokenUsage(input_tokens=7, output_tokens=3, total_tokens=10, calls=1)
        )

    monkeypatch.setattr(mod, 'generate_structured', fake)
    generator = mod.PersonaGenerator(model=MODEL, client=MagicMock())

    await generator.generate(agent_description='an agent', num_personas=1)
    await generator.generate(agent_description='an agent', num_personas=1)

    assert generator.get_usage().calls == 2
    assert generator.get_usage().total_tokens == 20
    generator.reset_usage()
    assert generator.get_usage().calls == 0


@pytest.mark.asyncio
async def test_trace_summarize_returns_the_usage_of_an_unusable_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summarize call that came back empty still billed — the usage comes back with it."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation import traces as traces_mod

    billed = TokenUsage(input_tokens=5, output_tokens=1, total_tokens=6, calls=1)

    async def fake(*_args: Any, **_kwargs: Any) -> StructuredResult[Any]:
        return StructuredResult(cast('Any', traces_mod._ConversationSummary(summary='  ')), '', billed)

    monkeypatch.setattr(traces_mod, 'generate_structured', fake)
    conversation = traces_mod.TraceConversation(trace_id='t-1', messages=[{'role': 'user', 'content': 'hi'}])

    summary, usage = await traces_mod._summarize_conversation(
        conversation,
        llm_client=MagicMock(),
        model=MODEL,
        config=traces_mod.TraceAnalysisConfig(),
    )

    assert summary is None
    assert usage == billed


@pytest.mark.asyncio
async def test_redteam_condense_returns_the_usage_of_the_call_it_made(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_condense_attack` hands its usage back so the recommendations phase can sum it."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.redteam.contracts import LLMConfig, RedTeamRecommendationConfig
    from evaluatorq.redteam.reports import recommendations as rec_mod

    billed = TokenUsage(input_tokens=9, output_tokens=4, total_tokens=13, calls=1)

    async def fake(*_args: Any, **_kwargs: Any) -> StructuredResult[Any]:
        return StructuredResult(cast('Any', rec_mod._CondensedAttackLLMResponse(analysis='it worked')), '', billed)

    monkeypatch.setattr(rec_mod, 'generate_structured', fake)
    result = MagicMock()

    block, usage = await rec_mod._condense_attack(
        'a very long attack block',
        result,
        RedTeamRecommendationConfig(),
        cast('Any', MagicMock()),
        MODEL,
        LLMConfig(),
        {},
        {},
    )

    assert 'it worked' in block
    assert usage == billed
