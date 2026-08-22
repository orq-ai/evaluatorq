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
from openai import LengthFinishReasonError
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


def _text_completion(
    content: str,
    *,
    prompt: int,
    completion: int,
    usage: bool = True,
    finish_reason: Any = 'stop',
) -> ChatCompletion:
    """A ``.create()`` response — rung 2/4's shape."""
    return ChatCompletion(
        id='cmpl-text',
        created=0,
        model=MODEL,
        object='chat.completion',
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,
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


# --- usage survives the raising paths too --------------------------------------


@pytest.mark.asyncio
async def test_usage_billed_before_a_raise_comes_back_on_the_exception() -> None:
    """The scenario the fold-on-return-only version lost entirely.

    Rung 1 answers unusably and rung 2 comes back truncated, which raises. Two
    provider calls were billed; before the exception carried them, both vanished
    past the ``return`` that folded ``usages`` and the phase logged "no usage
    reported by the provider".
    """
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(None, prompt=10, completion=1))
    client.chat.completions.create = AsyncMock(
        return_value=_text_completion('{"value": "half a str', prompt=20, completion=2, finish_reason='length')
    )

    with pytest.raises(RuntimeError, match='Raise the max_tokens budget') as error:
        await _generate(client)

    usage = getattr(error.value, 'usage', None)
    assert usage is not None
    assert usage.calls == 2
    assert usage.input_tokens == 30
    assert usage.output_tokens == 3


@pytest.mark.asyncio
async def test_a_truncated_parse_rung_bills_the_tokens_it_generated() -> None:
    """`LengthFinishReasonError` carries the completion it refused to parse, usage and all.

    This is the most common raising path and the most expensive one — the model
    generated the whole `max_tokens` budget. Reading the usage off the response
    is impossible here (the SDK raised instead of returning), so it comes off the
    exception; dropping it reported the priciest failure the ladder has as free.
    """
    register_model(
        MODEL,
        ModelInfo(input_cost_per_1k=1.0, output_cost_per_1k=2.0, provider='test', supports_responses=False),
    )
    truncated = _text_completion('{"value": "half a str', prompt=1000, completion=64, finish_reason='length')
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=LengthFinishReasonError(completion=truncated))
    client.chat.completions.create = AsyncMock()

    with pytest.raises(RuntimeError, match='Raise the max_tokens budget') as error:
        await _generate(client)

    usage = getattr(error.value, 'usage', None)
    assert usage is not None
    assert usage.calls == 1
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 64
    # Priced here rather than in the executor, which never returned.
    assert usage.priced_calls == 1
    assert usage.total_cost == pytest.approx(1000 / 1000 * 1.0 + 64 / 1000 * 2.0)
    # The ladder does not continue past a truncation.
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_truncated_parse_rung_without_usage_is_unpriced_not_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The ladder's "never as zero" policy applies to the raising path too."""
    truncated = _text_completion('{"value": "half', prompt=0, completion=0, usage=False, finish_reason='length')
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=LengthFinishReasonError(completion=truncated))

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match='Raise the max_tokens budget') as error:
        await _generate(client)

    usage = getattr(error.value, 'usage', None)
    assert usage is not None
    assert usage.calls == 1
    assert usage.priced_calls == 0
    assert usage.total_cost is None
    assert 'no readable usage block' in caplog.text


@pytest.mark.asyncio
async def test_a_provider_error_keeps_its_type_and_still_carries_the_usage() -> None:
    """The last rung's provider error must not be masked by a RuntimeError to carry usage.

    A caller that reads ``status_code`` off the exception is doing the right
    thing; the spend rides along as an attribute instead of a new type.
    """
    import httpx
    from openai import APIStatusError

    request = httpx.Request('POST', 'https://router.example/v3/router')
    # A 400 that is neither a schema nor a tool rejection: no rung degrades on
    # it and `with_retry` does not retry it, so it leaves the ladder as itself.
    context_length = APIStatusError(
        "This model's maximum context length is 8192 tokens",
        response=httpx.Response(400, request=request),
        body=None,
    )

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(None, prompt=10, completion=1))
    client.chat.completions.create = AsyncMock(
        side_effect=[_text_completion('not json', prompt=20, completion=2), context_length]
    )

    with pytest.raises(APIStatusError) as error:
        await _generate(client, extra_kwargs={'tools': [{'type': 'function', 'function': {'name': 'caller_tool'}}]})

    assert error.value.status_code == 400
    usage = getattr(error.value, 'usage', None)
    assert usage is not None
    # Rungs 1 and 2 billed; rung 3 was skipped (caller tools) and rung 4 raised.
    assert usage.calls == 2
    assert usage.total_tokens == 33


@pytest.mark.asyncio
async def test_a_refusal_carries_the_usage_of_the_call_that_refused() -> None:
    """The refusing rung billed as surely as one that answered."""
    client = MagicMock()
    refused = _parsed_completion(None, prompt=10, completion=1)
    refused.choices[0].message.refusal = 'no'
    client.chat.completions.parse = AsyncMock(return_value=refused)

    with pytest.raises(RuntimeError, match='model refused to generate') as error:
        await _generate(client)

    usage = getattr(error.value, 'usage', None)
    assert usage is not None
    assert usage.calls == 1
    assert usage.total_tokens == 11


@pytest.mark.asyncio
async def test_an_error_before_any_provider_call_carries_no_invented_usage() -> None:
    """Nothing billed means nothing to report — not a zero-token phantom call."""
    client = MagicMock()

    with pytest.raises(ValueError, match='structural') as error:
        await _generate(client, extra_kwargs={'response_format': {'type': 'json_object'}})

    assert getattr(error.value, 'usage', None) is None


# --- the reserved-key sets may be wider than contracts', never narrower --------


def test_structured_output_reserved_keys_derive_from_the_contracts_sets() -> None:
    """A key added to `contracts.py` must reach `generate_structured` on the same commit.

    The two sets here were restated rather than derived, so `contracts.py`'s
    claim that this module "imports them rather than keeping its own copies" was
    false and a new structural key would have been enforced on the executors but
    not on the ladder. They are supersets by construction now; this fails if
    someone re-hardcodes them.
    """
    from evaluatorq.common.structured_output import (
        _STRUCTURAL_KEYS,
        _STRUCTURAL_KEYS_BY_API,
        _STRUCTURAL_KEYS_RESPONSES,
    )
    from evaluatorq.contracts import _RESERVED_COMPLETION_KEYS, _RESERVED_RESPONSES_KEYS

    assert _RESERVED_COMPLETION_KEYS <= _STRUCTURAL_KEYS
    assert _RESERVED_RESPONSES_KEYS <= _STRUCTURAL_KEYS_RESPONSES
    # An api='responses' call can still fall through to the chat rungs, so its
    # reserved set is the union of both endpoints' — deliberately, not by accident.
    assert _STRUCTURAL_KEYS_BY_API['chat_completions'] == _STRUCTURAL_KEYS
    assert _STRUCTURAL_KEYS_BY_API['responses'] == _STRUCTURAL_KEYS | _STRUCTURAL_KEYS_RESPONSES
    # The extra breadth is the ladder's own, and is documented as such.
    assert {'max_completion_tokens'} <= _STRUCTURAL_KEYS
    assert {'text_format', 'max_output_tokens'} <= _STRUCTURAL_KEYS_RESPONSES
