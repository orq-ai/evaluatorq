from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
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
        model="openai/gpt-5.5",
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
