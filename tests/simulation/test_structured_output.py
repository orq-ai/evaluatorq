from __future__ import annotations

# ruff: noqa: S101
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

    with pytest.raises(RuntimeError, match="Raise the max_tokens budget"):
        await generate_structured(
            client,
            model="local-model",
            messages=[{"role": "user", "content": "return json"}],
            response_format=SampleResponse,
            temperature=0.0,
            max_tokens=4000,
            label="Sample.generate",
        )

    # No json_object fallback call — the truncated result is not salvaged.
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_structural_extra_kwargs_are_rejected() -> None:
    # extra_kwargs silently replacing response_format would defeat the schema
    # the helper exists to enforce — reserved keys raise instead (review fix).
    client = MagicMock()

    with pytest.raises(ValueError, match="structural"):
        await generate_structured(
            client,
            model="local-model",
            messages=[{"role": "user", "content": "return json"}],
            response_format=SampleResponse,
            max_tokens=100,
            label="Sample.generate",
            extra_kwargs={"response_format": {"type": "json_object"}},
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
    client.chat.completions.create = AsyncMock(
        return_value=_fallback_completion('{"value": "half a str', "length")
    )

    with pytest.raises(RuntimeError, match="Raise the max_tokens budget"):
        await generate_structured(
            client,
            model="local-model",
            messages=[{"role": "user", "content": "return json"}],
            response_format=SampleResponse,
            max_tokens=64,
            label="Sample.generate",
        )


@pytest.mark.asyncio
async def test_complete_fallback_content_is_returned() -> None:
    """The guard is on finish_reason, not on the content — a normal fallback still works."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_no_parsed_completion())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "ok"}', "stop"))

    parsed, raw = await generate_structured(
        client,
        model="local-model",
        messages=[{"role": "user", "content": "return json"}],
        response_format=SampleResponse,
        max_tokens=64,
        label="Sample.generate",
    )

    assert parsed is None
    assert raw == '{"value": "ok"}'


def _no_parsed_completion() -> MagicMock:
    """A parse() response with no validated model, which trips the fallback."""
    completion = MagicMock()
    completion.choices[0].message.refusal = None
    completion.choices[0].message.parsed = None
    return completion


def _schema_400() -> Any:
    """A 400 that reads as a structured-output-support problem."""
    import httpx
    from openai import APIStatusError

    request = httpx.Request("POST", "https://router.example/v3/router")
    return APIStatusError(
        "text.format is not supported",
        response=httpx.Response(400, request=request),
        body=None,
    )


@pytest.mark.asyncio
async def test_fallback_carries_the_schema_not_a_bare_json_object() -> None:
    """The fallback is what tells the model which keys to emit — json_object does not."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "ok"}', "stop"))

    _parsed, raw = await generate_structured(
        client,
        model="local-model",
        messages=[{"role": "user", "content": "return json"}],
        response_format=SampleResponse,
        max_tokens=64,
        label="Sample.generate",
    )

    assert raw == '{"value": "ok"}'
    await_args = client.chat.completions.create.await_args
    assert await_args is not None
    response_format = await_args.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "SampleResponse"
    assert response_format["json_schema"]["strict"] is False  # strict is what parse() just failed on
    assert "value" in response_format["json_schema"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_provider_rejecting_the_schema_form_still_gets_json_object() -> None:
    """Degrade, don't lose the call: a provider that refuses both forms keeps working."""
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=_schema_400())
    client.chat.completions.create = AsyncMock(
        side_effect=[_schema_400(), _fallback_completion('{"value": "ok"}', "stop")]
    )

    _parsed, raw = await generate_structured(
        client,
        model="local-model",
        messages=[{"role": "user", "content": "return json"}],
        response_format=SampleResponse,
        max_tokens=64,
        label="Sample.generate",
    )

    assert raw == '{"value": "ok"}'
    formats = [c.kwargs["response_format"] for c in client.chat.completions.create.await_args_list]
    assert [f["type"] for f in formats] == ["json_schema", "json_object"]


def _status_error(status: int, message: str) -> Any:
    """An APIStatusError carrying a body a caller can read the cause out of."""
    import httpx
    from openai import APIStatusError

    request = httpx.Request("POST", "https://router.example/v3/router")
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
    client.chat.completions.create = AsyncMock(return_value=_fallback_completion('{"value": "chat"}', "stop"))
    return client


async def _generate_via_responses(client: MagicMock, **kwargs: Any) -> tuple[Any, str]:
    return await generate_structured(
        client,
        model="local-model",
        messages=[{"role": "user", "content": "return json"}],
        response_format=SampleResponse,
        max_tokens=64,
        label="Sample.generate",
        api="responses",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_responses_leg_returns_the_parsed_model_without_touching_chat() -> None:
    """The happy path is one call on one endpoint — no chat span, no second bill."""
    client = _responses_client(_responses_result(SampleResponse(value="ok")))

    parsed, raw = await _generate_via_responses(client)

    assert parsed == SampleResponse(value="ok")
    assert raw == ""
    client.chat.completions.parse.assert_not_awaited()
    client.chat.completions.create.assert_not_awaited()
    client.responses.parse.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_responses_endpoint_degrades_to_the_chat_legs(caplog: pytest.LogCaptureFixture) -> None:
    """A provider without /responses keeps working — that is what makes api= safe to default on."""
    client = _responses_client(_status_error(404, "Unknown request URL: POST /v1/responses"))

    with caplog.at_level("WARNING"):
        _parsed, raw = await _generate_via_responses(client)

    assert raw == '{"value": "chat"}'
    client.chat.completions.parse.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()
    assert "HTTP 404" in caplog.text


@pytest.mark.asyncio
async def test_schema_rejection_degrades_to_the_chat_legs(caplog: pytest.LogCaptureFixture) -> None:
    """A 400 whose body names the schema form is a real capability signal."""
    client = _responses_client(_status_error(400, "text_format is not supported for this model"))

    with caplog.at_level("WARNING"):
        _parsed, raw = await _generate_via_responses(client)

    assert raw == '{"value": "chat"}'
    client.chat.completions.parse.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()
    assert "HTTP 400" in caplog.text


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

    with caplog.at_level("WARNING"):
        _parsed, raw = await _generate_via_responses(client)

    assert raw == '{"value": "chat"}'
    client.chat.completions.parse.assert_awaited_once()
    client.chat.completions.create.assert_awaited_once()
    assert "no output" in caplog.text


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

    with pytest.raises(RuntimeError, match="Raise the max_tokens budget"):
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
    client = _responses_client(_responses_result(SampleResponse(value="ok")))

    with pytest.raises(ValueError, match="structural"):
        await _generate_via_responses(client, extra_kwargs={"text_format": {"type": "json_object"}})

    client.responses.parse.assert_not_called()


@pytest.mark.asyncio
async def test_chat_structural_keys_stay_reserved_on_the_responses_path() -> None:
    """An api='responses' call can still reach the chat legs, so both key sets are reserved."""
    client = _responses_client(_responses_result(SampleResponse(value="ok")))

    with pytest.raises(ValueError, match="structural"):
        await _generate_via_responses(client, extra_kwargs={"messages": []})


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
