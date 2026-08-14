from __future__ import annotations

# ruff: noqa: S101
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import LengthFinishReasonError
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
