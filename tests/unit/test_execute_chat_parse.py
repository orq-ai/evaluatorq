from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError
from pydantic import BaseModel

from evaluatorq.common.llm_call import execute_chat_parse


class _V(BaseModel):
    value: bool
    explanation: str


def _completion(parsed: _V | None, refusal: str | None = None):
    msg = MagicMock()
    msg.parsed = parsed
    msg.refusal = refusal
    choice = MagicMock()
    choice.message = msg
    comp = MagicMock()
    comp.choices = [choice]
    comp.usage = None
    return comp


@pytest.mark.asyncio
async def test_parse_forwards_response_model_and_returns_completion():
    client = MagicMock()
    parsed = _V(value=True, explanation="ok")
    client.chat.completions.parse = AsyncMock(return_value=_completion(parsed))

    resp, usage = await execute_chat_parse(
        client=client,
        model="openai/gpt-5.4-mini",
        messages=[{"role": "user", "content": "hi"}],
        span=None,
        timeout_s=30.0,
        response_model=_V,
        temperature=None,
    )

    assert resp.choices[0].message.parsed is parsed
    kwargs = client.chat.completions.parse.call_args.kwargs
    assert kwargs["response_format"] is _V
    assert "temperature" not in kwargs  # None → omitted


@pytest.mark.asyncio
async def test_parse_includes_temperature_when_set():
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_completion(_V(value=False, explanation="x")))
    await execute_chat_parse(
        client=client, model="m", messages=[], span=None, timeout_s=5.0,
        response_model=_V, temperature=0.3,
    )
    assert client.chat.completions.parse.call_args.kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_parse_drops_reasoning_effort_and_retries_once():
    # A 400 mentioning "reasoning" causes reasoning_effort to be dropped and the call
    # retried exactly once; the retry succeeds and omits reasoning_effort.
    client = MagicMock()
    err = BadRequestError(
        message="unsupported parameter: reasoning_effort",
        response=MagicMock(status_code=400),
        body={"error": {"message": "reasoning_effort not supported"}},
    )
    client.chat.completions.parse = AsyncMock(
        side_effect=[err, _completion(_V(value=True, explanation="ok"))]
    )
    resp, _usage = await execute_chat_parse(
        client=client, model="m", messages=[], span=None, timeout_s=5.0,
        response_model=_V, extra_kwargs={"reasoning_effort": "high"},
    )
    assert resp.choices[0].message.parsed.value is True
    assert client.chat.completions.parse.call_count == 2
    assert "reasoning_effort" not in client.chat.completions.parse.call_args.kwargs
