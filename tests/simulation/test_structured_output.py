from __future__ import annotations

# ruff: noqa: S101
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIStatusError, LengthFinishReasonError
from pydantic import BaseModel

from evaluatorq.simulation.utils.structured_output import generate_structured


class SampleResponse(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_generate_structured_raises_when_parse_hits_length_limit() -> None:
    # Length-truncated structured output is unusable — fail with a clear,
    # actionable error instead of falling back to a same-budget json_object call
    # that would truncate again.
    parse_error = LengthFinishReasonError(completion=MagicMock())

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=parse_error)
    client.chat.completions.create = AsyncMock()

    with pytest.raises(RuntimeError, match='Raise the max_tokens budget'):
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            temperature=0.0,
            max_tokens=4000,
            label='Sample.generate',
        )

    # No json_object fallback call — the truncated result is not salvaged.
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_structural_extra_kwargs_are_rejected() -> None:
    # extra_kwargs silently replacing response_format would defeat the schema
    # the helper exists to enforce — reserved keys raise instead (review fix).
    client = MagicMock()

    with pytest.raises(ValueError, match='structural'):
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            max_tokens=100,
            label='Sample.generate',
            extra_kwargs={'response_format': {'type': 'json_object'}},
        )

    client.chat.completions.parse.assert_not_called()


def _fallback_completion(content: str, finish_reason: str) -> MagicMock:
    completion = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_generate_structured_raises_when_the_fallback_hits_the_length_limit() -> None:
    # The SDK raises LengthFinishReasonError for us on the parse() leg but not on
    # the json_object one, where a cut-off body comes back looking like ordinary
    # content — extract_json_from_response salvages half an object and the caller
    # scores a half-answer. Same budget, same defect, same loud failure.
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "half a str', 'length'))

    with pytest.raises(RuntimeError, match='Raise the max_tokens budget'):
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            max_tokens=64,
            label='Sample.generate',
        )


@pytest.mark.asyncio
async def test_complete_fallback_content_is_returned() -> None:
    """The guard is on finish_reason, not on the content — a normal fallback still works."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "ok"}', 'stop'))

    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )
    parsed, raw = result.parsed, result.raw

    assert parsed is not None
    assert parsed.value == 'ok'
    assert raw == '{"value": "ok"}'


@pytest.mark.asyncio
async def test_fenced_fallback_content_is_parsed_by_the_helper() -> None:
    """A provider on a text rung fences its payload; the ladder salvages it so every
    caller gets a model, not a raw string it has to fence-strip itself."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(
        return_value=_fallback_completion('Here you go:\n```json\n{"value": "ok"}\n```', 'stop')
    )

    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )
    parsed, raw = result.parsed, result.raw

    assert parsed is not None
    assert parsed.value == 'ok'
    assert '```json' in raw  # raw stays untouched for callers with their own salvage


@pytest.mark.asyncio
async def test_unparseable_fallback_content_returns_none_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Off-schema content must not come back as a parsed model — None plus a warning."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('sorry, no json here', 'stop'))

    with caplog.at_level(logging.WARNING):
        result = await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            max_tokens=64,
            label='Sample.generate',
        )
    parsed, raw = result.parsed, result.raw

    assert parsed is None
    assert raw == 'sorry, no json here'
    assert 'did not validate' in caplog.text


async def _run_to_the_tool_rung(client: MagicMock) -> tuple[Any, str]:
    """Drive the ladder with rung 2 answering nothing usable, so rung 3 runs."""
    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )
    return result.parsed, result.raw


@pytest.mark.asyncio
async def test_forced_tool_call_rung_answers_when_json_schema_did_not() -> None:
    """Rung 3 is the strict backup: tool_choice leaves the model no prose channel."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=[_no_parsed_completion(), _tool_completion('{"value": "ok"}')]
    )
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))

    parsed, raw = await _run_to_the_tool_rung(client)

    assert parsed is not None
    assert parsed.value == 'ok'
    assert raw == '{"value": "ok"}'
    # Rung 4 never ran — the tool rung answered first.
    assert client.chat.completions.create.await_count == 1

    tool_kwargs = client.chat.completions.parse.await_args_list[1].kwargs
    assert tool_kwargs['tool_choice'] == {'type': 'function', 'function': {'name': 'SampleResponse'}}
    assert tool_kwargs['tools'][0]['function']['name'] == 'SampleResponse'


@pytest.mark.asyncio
async def test_forced_tool_call_appends_a_nudge_without_mutating_the_caller_messages() -> None:
    """The nudge rescues providers that downgrade a named tool_choice to auto — but the
    caller's list is theirs, and the appended turn must be visible in the sent params."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=[_no_parsed_completion(), _tool_completion('{"value": "ok"}')]
    )
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))
    messages = [{'role': 'user', 'content': 'return json'}]

    await generate_structured(
        client,
        model='local-model',
        messages=messages,
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )

    assert messages == [{'role': 'user', 'content': 'return json'}]  # untouched
    sent = client.chat.completions.parse.await_args_list[1].kwargs['messages']
    assert len(sent) == 2
    assert sent[-1]['role'] == 'user'
    assert 'SampleResponse' in sent[-1]['content']
    # Rung 1 got the caller's prompt verbatim — only rung 3 edits it.
    assert client.chat.completions.parse.await_args_list[0].kwargs['messages'] == messages


@pytest.mark.asyncio
async def test_forced_tool_call_rung_is_skipped_when_the_caller_passes_tools() -> None:
    """A caller's tools are functional; forcing ours would break the call this rung
    only exists to salvage. Skip to json_object instead."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _fallback_completion('not json at all', 'stop'),
            _fallback_completion('{"value": "ok"}', 'stop'),
        ]
    )

    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        extra_kwargs={'tools': [{'type': 'function', 'function': {'name': 'caller_tool'}}]},
    )
    parsed = result.parsed

    assert parsed is not None
    assert parsed.value == 'ok'
    assert client.chat.completions.parse.await_count == 1  # rung 1 only; rung 3 skipped
    assert client.chat.completions.create.await_count == 2  # rungs 2 and 4


@pytest.mark.asyncio
async def test_forced_tool_call_without_tool_calls_falls_through_to_json_object() -> None:
    """A model that ignores tool_choice and answers in prose must not end the ladder."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _fallback_completion('not json at all', 'stop'),
            _fallback_completion('{"value": "ok"}', 'stop'),
        ]
    )

    parsed, _raw = await _run_to_the_tool_rung(client)

    assert parsed is not None
    assert parsed.value == 'ok'
    assert client.chat.completions.parse.await_count == 2  # rung 3 was attempted


@pytest.mark.asyncio
async def test_tool_arguments_are_validated_by_the_helper_when_the_sdk_did_not() -> None:
    """Non-OpenAI providers come back without parsed_arguments; the raw argument
    string still goes through the shared salvage rather than being trusted."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=[_no_parsed_completion(), _tool_completion('```json\n{"value": "ok"}\n```')]
    )
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))

    parsed, _raw = await _run_to_the_tool_rung(client)

    assert parsed is not None
    assert parsed.value == 'ok'


@pytest.mark.asyncio
async def test_total_failure_returns_the_last_non_empty_raw() -> None:
    """Nothing validated anywhere — the caller still gets text to log."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _fallback_completion('first attempt', 'stop'),
            _fallback_completion('last attempt', 'stop'),
        ]
    )

    parsed, raw = await _run_to_the_tool_rung(client)

    assert parsed is None
    assert raw == 'last attempt'


def _no_parsed_completion() -> MagicMock:
    """A parse() response with no validated model, which trips the fallback.

    tool_calls is emptied too: on the forced-tool rung a bare MagicMock would
    auto-create a truthy parsed_arguments and fake a tool answer.
    """
    completion = MagicMock()
    completion.choices[0].message.refusal = None
    completion.choices[0].message.parsed = None
    completion.choices[0].message.tool_calls = []
    completion.choices[0].finish_reason = 'stop'
    return completion


def _tool_completion(arguments: str, *, parsed_arguments: Any = None) -> MagicMock:
    """A parse() response answering through a forced tool call."""
    completion = MagicMock()
    choice = MagicMock()
    choice.finish_reason = 'tool_calls'
    choice.message.refusal = None
    choice.message.parsed = None
    call = MagicMock()
    call.function.arguments = arguments
    call.function.parsed_arguments = parsed_arguments
    choice.message.tool_calls = [call]
    completion.choices = [choice]
    return completion


def _schema_400() -> Any:
    """A 400 that reads as a structured-output-support problem."""
    import httpx
    from openai import APIStatusError

    request = httpx.Request('POST', 'https://router.example/v3/router')
    return APIStatusError(
        'text.format is not supported',
        response=httpx.Response(400, request=request),
        body=None,
    )


@pytest.mark.asyncio
async def test_fallback_carries_the_schema_not_a_bare_json_object() -> None:
    """The fallback is what tells the model which keys to emit — json_object does not."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "ok"}', 'stop'))

    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )
    _parsed, raw = result.parsed, result.raw

    assert raw == '{"value": "ok"}'
    await_args = client.chat.completions.create.await_args
    assert await_args is not None
    response_format = await_args.kwargs['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['name'] == 'SampleResponse'
    assert response_format['json_schema']['strict'] is False  # strict is what parse() just failed on
    assert 'value' in response_format['json_schema']['schema']['properties']


@pytest.mark.asyncio
async def test_provider_rejecting_the_schema_form_still_gets_json_object() -> None:
    """Degrade, don't lose the call: a provider that refuses both forms keeps working."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    client.chat.completions.create = AsyncMock(
        side_effect=[_schema_400(), _fallback_completion('{"value": "ok"}', 'stop')]
    )

    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )
    _parsed, raw = result.parsed, result.raw

    assert raw == '{"value": "ok"}'
    formats = [c.kwargs['response_format'] for c in client.chat.completions.create.await_args_list]
    assert [f['type'] for f in formats] == ['json_schema', 'json_object']


def _status_error(status: int, message: str) -> Any:
    """An APIStatusError carrying a body a caller can read the cause out of."""
    import httpx
    from openai import APIStatusError

    request = httpx.Request('POST', 'https://router.example/v3/router')
    return APIStatusError(message, response=httpx.Response(status, request=request), body=None)


def _responses_result(parsed: Any, incomplete_reason: str | None = None) -> MagicMock:
    response = MagicMock()
    response.output_parsed = parsed
    response.output_text = '' if parsed is None else parsed.model_dump_json()
    response.stop_reason = None
    response.incomplete_details = None if incomplete_reason is None else MagicMock(reason=incomplete_reason)
    _set_response_output(response, response.output_text)
    return response


def _set_response_output(response: MagicMock, text: str) -> None:
    response.output_text = text
    if not text:
        response.output = []
        return
    content = SimpleNamespace(
        type='output_text',
        text=text,
        to_dict=lambda: {'type': 'output_text', 'text': text},
    )
    response.output = [
        SimpleNamespace(
            type='message',
            content=[content],
            to_dict=lambda: {'type': 'message', 'content': [content.to_dict()]},
        )
    ]


def _responses_refusal(reason: str) -> MagicMock:
    response = _responses_result(None)
    response.output = [MagicMock(content=[MagicMock(type='refusal', refusal=reason)])]
    return response


def _responses_client(parse_result: Any) -> MagicMock:
    """A client whose Responses leg is driven by the caller and whose chat legs succeed."""
    client = MagicMock()
    if isinstance(parse_result, BaseException):
        client.responses.create = AsyncMock(side_effect=parse_result)
    else:
        client.responses.create = AsyncMock(return_value=parse_result)
    client.responses.parse = AsyncMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "chat"}', 'stop'))
    return client


async def _generate_via_responses(client: MagicMock, **kwargs: Any) -> tuple[Any, str]:
    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        api='responses',
        **kwargs,
    )
    return result.parsed, result.raw


@pytest.mark.asyncio
async def test_responses_leg_returns_the_parsed_model_without_touching_chat() -> None:
    """The happy path is one call on one endpoint — no chat span, no second bill."""
    client = _responses_client(_responses_result(SampleResponse(value='ok')))

    parsed, raw = await _generate_via_responses(client)

    assert parsed == SampleResponse(value='ok')
    assert raw == ''
    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()
    client.responses.parse.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_responses_endpoint_degrades_to_the_chat_legs(caplog: pytest.LogCaptureFixture) -> None:
    """A provider without /responses keeps working — that is what makes api= safe to default on."""
    client = _responses_client(_status_error(404, 'Unknown request URL: POST /v1/responses'))

    with caplog.at_level('WARNING'):
        _parsed, raw = await _generate_via_responses(client)

    assert raw == '{"value": "chat"}'
    client.chat.completions.parse.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()
    assert 'HTTP 404' in caplog.text


@pytest.mark.asyncio
async def test_schema_rejection_degrades_to_the_chat_legs(caplog: pytest.LogCaptureFixture) -> None:
    """A 400 whose body names the schema form is a real capability signal."""
    client = _responses_client(_status_error(400, 'text_format is not supported for this model'))

    with caplog.at_level('WARNING'):
        _parsed, raw = await _generate_via_responses(client)

    assert raw == '{"value": "chat"}'
    client.chat.completions.parse.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()
    assert 'HTTP 400' in caplog.text


@pytest.mark.asyncio
async def test_unrelated_400_raises_instead_of_being_blamed_on_the_provider() -> None:
    # A bad parameter or an over-length context is not "Responses unsupported".
    # Degrading on it would log a false cause and re-bill the same broken
    # request on the chat leg, where it fails again with the real error.
    client = _responses_client(_status_error(400, "This model's maximum context length is 8192 tokens"))

    with pytest.raises(APIStatusError):
        await _generate_via_responses(client)

    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_text_unsupported_400_is_not_classified_as_schema_rejection() -> None:
    client = _responses_client(_status_error(400, 'the text parameter is not supported by this deployment'))

    with pytest.raises(APIStatusError):
        await _generate_via_responses(client)

    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_unparsed_responses_output_degrades_to_the_chat_legs(caplog: pytest.LogCaptureFixture) -> None:
    """No exception and no truncation, but nothing parsed — take the answer from chat."""
    client = _responses_client(_responses_result(None))

    with caplog.at_level('WARNING'):
        _parsed, raw = await _generate_via_responses(client)

    assert raw == '{"value": "chat"}'
    client.chat.completions.parse.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()
    assert 'no output' in caplog.text


@pytest.mark.asyncio
async def test_responses_refusal_does_not_degrade_to_chat() -> None:
    client = _responses_client(_responses_refusal('safety refusal'))

    with pytest.raises(RuntimeError, match='model refused to generate: safety refusal'):
        await _generate_via_responses(client)

    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_truncated_responses_output_raises_rather_than_degrading() -> None:
    # Same rule as the chat leg: the chat fallback would run at the same budget
    # and truncate again, so a cut-off payload fails loudly and actionably.
    response = _responses_result(None)
    _set_response_output(response, '{"value": "half a str')
    response.stop_reason = 'length'
    client = _responses_client(response)

    with pytest.raises(RuntimeError, match='Raise the max_tokens budget'):
        await _generate_via_responses(client)

    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_nested_truncation_metadata_wins_over_a_non_length_stop_reason() -> None:
    response = _responses_result(None, incomplete_reason='max_output_tokens')
    _set_response_output(response, '{"value": "half a str')
    response.stop_reason = 'stop'
    client = _responses_client(response)

    with pytest.raises(RuntimeError, match='Raise the max_tokens budget'):
        await _generate_via_responses(client)


@pytest.mark.asyncio
async def test_responses_structural_extra_kwargs_are_rejected() -> None:
    # text is the Responses leg's name for the schema this helper exists
    # to enforce; extra_kwargs replacing it would return a well-formed object of
    # the wrong type through the cast, with no error until a field access fails.
    client = _responses_client(_responses_result(SampleResponse(value='ok')))

    with pytest.raises(ValueError, match='structural'):
        await _generate_via_responses(client, extra_kwargs={'text_format': {'type': 'json_object'}})

    client.responses.parse.assert_not_called()


@pytest.mark.asyncio
async def test_chat_structural_keys_stay_reserved_on_the_responses_path() -> None:
    """An api='responses' call can still reach the chat legs, so both key sets are reserved."""
    client = _responses_client(_responses_result(SampleResponse(value='ok')))

    with pytest.raises(ValueError, match='structural'):
        await _generate_via_responses(client, extra_kwargs={'messages': []})


@pytest.mark.asyncio
async def test_malformed_responses_payload_is_not_assumed_to_be_truncated() -> None:
    response = _responses_result(None)
    _set_response_output(response, '{"value": "half a str')
    response.stop_reason = 'stop'
    client = _responses_client(response)

    with pytest.raises(RuntimeError, match='did not validate against SampleResponse'):
        await _generate_via_responses(client)

    # The payload is invalid, but there is no provider truncation signal.
    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_schema_validation_error_is_not_misreported_as_truncation() -> None:
    response = _responses_result(None)
    _set_response_output(response, '{"other": "ok"}')
    response.stop_reason = 'stop'
    client = _responses_client(response)

    with pytest.raises(RuntimeError, match='did not validate against SampleResponse') as error:
        await _generate_via_responses(client)

    assert 'Raise the max_tokens budget' not in str(error.value)
    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_max_completion_tokens_extra_kwargs_is_rejected_on_the_chat_path() -> None:
    """F3: the chat leg used to omit its own token-cap field from the reserved
    set while the Responses leg reserved max_output_tokens — letting
    extra_kwargs silently override the chat completion budget but not the
    Responses one. Both must be reserved now.
    """
    client = MagicMock()

    with pytest.raises(ValueError, match='structural'):
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            max_tokens=100,
            label='Sample.generate',
            extra_kwargs={'max_completion_tokens': 1},
        )

    client.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_reasoning_effort_reaches_the_chat_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    """F4: reasoning_effort must reach the request the chat legs actually send."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        return_value=_no_parsed_completion()  # falls through past rung 1
    )
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "chat"}', 'stop'))

    await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        reasoning_effort='high',
    )

    await_args = client.chat.completions.parse.await_args
    assert await_args is not None
    assert await_args.kwargs['reasoning_effort'] == 'high'


@pytest.mark.asyncio
async def test_reasoning_effort_and_timeout_reach_the_responses_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    """F4: reasoning_effort and a caller-supplied timeout_s must reach
    execute_response, not just be accepted and silently dropped.
    """
    import evaluatorq.common.structured_output as structured_output_module

    captured: dict[str, Any] = {}

    async def _fake_execute_response(**kwargs: Any) -> tuple[Any, None]:
        captured.update(kwargs)
        return _responses_result(SampleResponse(value='ok')), None

    monkeypatch.setattr(structured_output_module, 'execute_response', _fake_execute_response)

    await generate_structured(
        MagicMock(),
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        api='responses',
        reasoning_effort='low',
        timeout_s=17.0,
    )

    assert captured['reasoning_effort'] == 'low'
    assert captured['timeout_s'] == 17.0


@pytest.mark.asyncio
async def test_extra_body_reaches_the_responses_request() -> None:
    """The router body must reach the Responses request, not trip the guard.

    `extra_body` is reserved inside `extra_kwargs` on both endpoints, so folding
    the dedicated parameter into that dict made every `api='responses'` call with
    a router body raise `ValueError` in `execute_response` instead of sending it.
    """
    client = MagicMock()
    captured: dict[str, object] = {}

    async def _create(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _StopAfterCapture

    client.responses.create = AsyncMock(side_effect=_create)

    with pytest.raises(_StopAfterCapture):
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            max_tokens=64,
            label='Sample.generate',
            api='responses',
            extra_body={'retry': {'count': 3}},
        )

    assert captured['extra_body'] == {'retry': {'count': 3}}


class _StopAfterCapture(Exception):
    """Ends the call once the request kwargs have been recorded."""


# --- the chat rungs run through common.llm_call, so they inherit its repairs ---


def _reasoning_400() -> Any:
    """The 400 a non-reasoning model answers `reasoning_effort` with."""
    import httpx
    from openai import BadRequestError

    request = httpx.Request('POST', 'https://router.example/v3/router')
    return BadRequestError(
        "Unsupported parameter: 'reasoning_effort' is not supported with this model.",
        response=httpx.Response(400, request=request),
        body=None,
    )


def _parsed_completion(value: str) -> MagicMock:
    """A parse() response the SDK validated into the requested model."""
    completion = MagicMock()
    choice = MagicMock()
    choice.finish_reason = 'stop'
    choice.message.refusal = None
    choice.message.parsed = SampleResponse(value=value)
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_chat_rung_survives_a_model_rejecting_reasoning_effort() -> None:
    """The acceptance test for routing the rungs through `common.llm_call`.

    `reasoning_effort` is injected by this helper but the drop-and-retry that
    handles a model rejecting it lives in the executors. While the rungs called
    ``client.chat.completions.parse`` directly, that 400 escaped rung 1 — its
    body names `reasoning_effort`, not the schema, so the schema-rejection
    fall-through did not catch it — and killed the whole structured call, while
    the `api='responses'` leg one function away self-healed.
    """
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=[_reasoning_400(), _parsed_completion('ok')])
    client.chat.completions.create = AsyncMock()

    result = await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        reasoning_effort='high',
    )

    assert result.parsed is not None
    assert result.parsed.value == 'ok'
    # Rung 1 answered on the retry — the ladder never had to degrade.
    client.chat.completions.create.assert_not_awaited()
    first, retried = client.chat.completions.parse.await_args_list
    assert first.kwargs['reasoning_effort'] == 'high'
    assert 'reasoning_effort' not in retried.kwargs


@pytest.mark.asyncio
async def test_a_later_call_strips_the_rejected_reasoning_effort_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The memo must not degrade the call silently.

    Once a model is memoized as a rejector the parameter is dropped before the
    request, so the user's configured effort is not in force. That is worth one
    warning per (endpoint, model, tools) key — and only one, since a per-call
    warning across a long run is noise.
    """
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=[_reasoning_400(), _parsed_completion('ok'), _parsed_completion('ok'), _parsed_completion('ok')]
    )
    client.chat.completions.create = AsyncMock()

    async def _call() -> None:
        await generate_structured(
            client,
            model='local-model',
            messages=[{'role': 'user', 'content': 'return json'}],
            response_format=SampleResponse,
            max_tokens=64,
            label='Sample.generate',
            reasoning_effort='high',
        )

    with caplog.at_level(logging.DEBUG, logger='evaluatorq.common.llm_call'):
        await _call()  # pays the 400 once
        await _call()  # first up-front strip: warns
        await _call()  # later strips: debug

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and 'stripping it up front' in r.message]
    assert len(warnings) == 1, 'the first silent downgrade must be announced exactly once'
    assert any(r.levelno == logging.DEBUG and 'stripped again' in r.message for r in caplog.records)
    # And the parameter really is gone from the later requests.
    for call in client.chat.completions.parse.await_args_list[1:]:
        assert 'reasoning_effort' not in call.kwargs


@pytest.mark.asyncio
async def test_timeout_s_bounds_the_chat_rungs_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """`timeout_s` used to bound only the Responses leg; the chat rungs take it now."""
    import evaluatorq.common.structured_output as structured_output_module

    captured: list[dict[str, Any]] = []

    async def _fake_execute_chat_parse(**kwargs: Any) -> tuple[Any, None]:
        captured.append(kwargs)
        return _parsed_completion('ok'), None

    monkeypatch.setattr(structured_output_module, 'execute_chat_parse', _fake_execute_chat_parse)

    await generate_structured(
        MagicMock(),
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        timeout_s=23.0,
    )

    assert captured[0]['timeout_s'] == 23.0
    assert captured[0]['max_completion_tokens'] == 64


@pytest.mark.asyncio
async def test_extra_kwargs_still_win_over_a_rungs_own_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """The executors apply extra_kwargs last, which is the order the ladder had."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion('ok'))

    await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        temperature=0.0,
        extra_kwargs={'temperature': 0.9, 'top_p': 0.5},
    )

    await_args = client.chat.completions.parse.await_args
    assert await_args is not None
    sent = await_args.kwargs
    assert sent['temperature'] == 0.9, "the caller's option wins over the ladder's base field"
    assert sent['top_p'] == 0.5


@pytest.mark.asyncio
async def test_extra_body_reaches_the_chat_rungs() -> None:
    """The router body rides the executors' dedicated parameter, not extra_kwargs."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion('ok'))

    await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
        extra_body={'retry': {'count': 3}},
    )

    await_args = client.chat.completions.parse.await_args
    assert await_args is not None
    assert await_args.kwargs['extra_body'] == {'retry': {'count': 3}}


class _FakeSpan:
    """Records `set_attribute` calls so a rung's span attributes can be asserted on."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


def _capture_span(monkeypatch: pytest.MonkeyPatch) -> _FakeSpan:
    """Give the ladder one recording span in place of the (absent) real tracer."""
    import contextlib

    import evaluatorq.common.structured_output as structured_output_module

    span = _FakeSpan()

    @contextlib.asynccontextmanager
    async def _fake_span(**_kwargs: Any) -> AsyncIterator[_FakeSpan]:
        yield span

    monkeypatch.setattr(structured_output_module, 'with_llm_span', _fake_span)
    return span


@pytest.mark.asyncio
async def test_the_forced_tool_nudge_is_recorded_on_the_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rung 3's docstring claims the appended turn is visible on the ladder's span.

    The rung routes through `execute_chat_parse`, which records the messages it
    sends, so the nudged list reaches `gen_ai.input.messages` — and rung 3 also
    writes it to its own attribute, which is what the next test needs.
    """
    import evaluatorq.common.llm_call as llm_call_module

    recorded: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(llm_call_module, 'record_llm_input', lambda _span, messages: recorded.append(messages))
    span = _capture_span(monkeypatch)

    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=[_no_parsed_completion(), _tool_completion('{"value": "ok"}')]
    )
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))

    await _run_to_the_tool_rung(client)

    assert recorded[-1][-1]['role'] == 'user'
    assert 'SampleResponse' in recorded[-1][-1]['content']
    assert 'SampleResponse' in span.attributes['orq.structured_output.tool_nudge']


@pytest.mark.asyncio
async def test_the_nudge_survives_a_rung_4_that_runs_after_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """`record_llm_input` *sets* the input attribute, so rung 4 overwrites rung 3's list.

    That is precisely the run a reader cares about — rung 3 failing is why rung 4
    ran — so the nudge has its own attribute, which no later rung touches.
    """
    import evaluatorq.common.llm_call as llm_call_module

    recorded: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(llm_call_module, 'record_llm_input', lambda _span, messages: recorded.append(messages))
    span = _capture_span(monkeypatch)

    client = MagicMock()
    # Rung 3 answers without a tool call, so rung 4 runs and answers.
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _fallback_completion('not json at all', 'stop'),
            _fallback_completion('{"value": "ok"}', 'stop'),
        ]
    )

    parsed, _raw = await _run_to_the_tool_rung(client)

    assert parsed is not None  # rung 4 answered after rung 3
    # The recorded input no longer carries the nudge: rung 4 replaced it.
    assert len(recorded[-1]) == 1
    assert 'SampleResponse' not in recorded[-1][-1]['content']
    # The dedicated attribute still does.
    assert 'SampleResponse' in span.attributes['orq.structured_output.tool_nudge']
    assert span.attributes['orq.structured_output.leg'] == 'json_object'


# --- every degraded exit announces itself -------------------------------------


@pytest.mark.asyncio
async def test_an_exhausted_ladder_says_it_ran_out_of_rungs(caplog: pytest.LogCaptureFixture) -> None:
    """Four rungs, up to five billed calls, and nothing to show: that is worth a line."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))

    with caplog.at_level(logging.WARNING):
        result = await _run_to_the_tool_rung(client)

    assert result[0] is None
    assert 'none produced output validating against SampleResponse' in caplog.text


@pytest.mark.asyncio
async def test_empty_content_warns_like_off_schema_content_does(caplog: pytest.LogCaptureFixture) -> None:
    """Two neighbouring branches must not differ in whether they log."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('', 'stop'))

    with caplog.at_level(logging.WARNING):
        await _run_to_the_tool_rung(client)

    assert 'returned empty content' in caplog.text


@pytest.mark.asyncio
async def test_a_response_with_no_choices_warns_rather_than_vanishing(caplog: pytest.LogCaptureFixture) -> None:
    """An empty `choices` list used to return an empty result with no explanation."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    no_choices = MagicMock()
    no_choices.choices = []
    client.chat.completions.create = AsyncMock(return_value=no_choices)

    with caplog.at_level(logging.WARNING):
        await _run_to_the_tool_rung(client)

    assert 'returned no choices' in caplog.text


@pytest.mark.asyncio
async def test_the_parse_rung_with_no_choices_degrades_instead_of_raising_indexerror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rung 1 indexed `choices[0]` unguarded while both neighbours warned and fell through.

    A provider answering with an empty `choices` list took the whole ladder down
    with an IndexError instead of degrading to rung 2.
    """
    no_choices = MagicMock()
    no_choices.choices = []
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=[no_choices, _tool_completion('{"value": "ok"}')])
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))

    with caplog.at_level(logging.WARNING):
        parsed, _raw = await _run_to_the_tool_rung(client)

    assert parsed is not None  # the ladder continued and rung 3 answered
    assert 'parse() returned no choices' in caplog.text


@pytest.mark.asyncio
async def test_a_tool_rung_with_no_choices_warns_too(caplog: pytest.LogCaptureFixture) -> None:
    no_choices = MagicMock()
    no_choices.choices = []
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=[_no_parsed_completion(), no_choices])
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('not json at all', 'stop'))

    with caplog.at_level(logging.WARNING):
        await _run_to_the_tool_rung(client)

    assert 'forced tool call returned no choices' in caplog.text


# --- one retry layer ----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_injected_clients_sdk_retries_are_disarmed_before_the_ladder_runs() -> None:
    """Every rung is wrapped in `with_retry`, so the SDK budget must be zeroed first.

    Left armed, the two layers multiply — five outer attempts over a client doing
    two SDK retries is fifteen requests per rung. The client is cloned, never
    mutated: it belongs to the caller.
    """
    injected = MagicMock()
    injected.max_retries = 2
    disarmed = MagicMock()
    disarmed.max_retries = 0
    disarmed.chat.completions.parse = AsyncMock(return_value=_parsed_completion('ok'))
    injected.with_options = MagicMock(return_value=disarmed)

    result = await generate_structured(
        injected,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )

    assert result.parsed is not None
    injected.with_options.assert_called_once_with(max_retries=0)
    # The disarmed clone is the one that talked to the provider, and the caller's
    # own client was not reconfigured in place.
    disarmed.chat.completions.parse.assert_awaited_once()
    injected.chat.completions.parse.assert_not_called()
    assert injected.max_retries == 2


@pytest.mark.asyncio
async def test_a_client_without_an_integer_retry_budget_is_left_alone() -> None:
    """Test doubles (and clients already at 0) must not be cloned into something else."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion('ok'))

    await generate_structured(
        client,
        model='local-model',
        messages=[{'role': 'user', 'content': 'return json'}],
        response_format=SampleResponse,
        max_tokens=64,
        label='Sample.generate',
    )

    client.with_options.assert_not_called()
    client.chat.completions.parse.assert_awaited_once()
