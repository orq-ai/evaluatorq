from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError
from pydantic import BaseModel, create_model

from evaluatorq.common.judge import run_judge
from evaluatorq.contracts import LLMCallConfig


def _cfg() -> LLMCallConfig:
    return LLMCallConfig(model="openai/gpt-5.5", temperature=1.0, max_tokens=8000, timeout_ms=30000)


def _verdict_model() -> type[BaseModel]:
    return create_model("Verdict", value=(bool, ...), explanation=(str, ...))


def _parsed_completion(value, explanation, refusal=None):
    msg = MagicMock()
    msg.parsed = None if refusal else _verdict_model()(value=value, explanation=explanation)
    msg.refusal = refusal
    choice = MagicMock(); choice.message = msg
    comp = MagicMock(); comp.choices = [choice]; comp.usage = None
    return comp


@pytest.mark.asyncio
async def test_tier1_parse_normalizes_to_payload():
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(True, "resisted"))
    out = await run_judge(
        client=client, model="openai/gpt-5.5", cfg=_cfg(),
        prompt_template="judge: {{output.response}}", replacements={"output.response": "hi"},
        system_prompt="sys", response_model=_verdict_model(),
    )
    assert out.error_kind is None
    assert out.payload is not None
    assert out.payload.value is True
    assert out.payload.explanation == "resisted"


@pytest.mark.asyncio
async def test_tier1_refusal_maps_to_abstain():
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(None, "", refusal="I cannot."))
    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="s", response_model=_verdict_model(),
    )
    assert out.payload is not None
    assert out.payload.abstain is True
    assert out.payload.value is None


@pytest.mark.asyncio
async def test_fallback_to_json_object_on_badrequest():
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=BadRequestError(
            message="response_format json_schema not supported",
            response=MagicMock(status_code=400), body={"error": {"message": "json_schema unsupported"}},
        )
    )
    create_comp = MagicMock()
    create_comp.choices = [MagicMock(message=MagicMock(content='{"value": true, "explanation": "ok"}'))]
    create_comp.usage = None
    client.chat.completions.create = AsyncMock(return_value=create_comp)

    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="sys", response_model=_verdict_model(),
    )
    assert out.payload is not None
    assert out.payload.value is True
    # fallback injected the schema into the system prompt
    sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert any("value" in m["content"] for m in sent_messages if m["role"] == "system")


@pytest.mark.asyncio
async def test_response_model_none_uses_create_unchanged():
    client = MagicMock()
    comp = MagicMock()
    comp.choices = [MagicMock(message=MagicMock(content='{"value": false, "explanation": "x"}'))]
    comp.usage = None
    client.chat.completions.create = AsyncMock(return_value=comp)

    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="sys",
    )
    assert client.chat.completions.create.called
    assert not hasattr(client.chat.completions, "parse") or not client.chat.completions.parse.called
    assert out.payload is not None
    assert out.payload.value is False
    # default json_object response_format preserved
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_temperature_none_override_omits_temperature():
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(return_value=_parsed_completion(True, "ok"))
    await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="s",
        response_model=_verdict_model(), temperature=None,
    )
    assert "temperature" not in client.chat.completions.parse.call_args.kwargs
