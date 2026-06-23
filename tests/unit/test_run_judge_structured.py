from __future__ import annotations

import asyncio
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import BadRequestError
from pydantic import BaseModel, create_model

from evaluatorq.common.judge import JudgeError, run_judge
from evaluatorq.contracts import LLMCallConfig


def _cfg() -> LLMCallConfig:
    return LLMCallConfig(model="openai/gpt-5.4-mini", temperature=1.0, max_tokens=8000, timeout_ms=30000)


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
        client=client, model="openai/gpt-5.4-mini", cfg=_cfg(),
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
async def test_structured_output_false_still_injects_schema():
    # structured_output=False must NOT silently drop the verdict schema: it routes
    # through the json_object path with the schema injected into the system prompt,
    # so a verdict model (e.g. a label set) still constrains the model.
    client = MagicMock()
    comp = MagicMock()
    comp.choices = [MagicMock(message=MagicMock(content='{"value": true, "explanation": "ok"}'))]
    comp.usage = None
    client.chat.completions.create = AsyncMock(return_value=comp)
    client.chat.completions.parse = AsyncMock()  # must not be called

    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="sys",
        response_model=_verdict_model(), structured_output=False,
    )
    assert out.payload is not None
    assert out.payload.value is True
    assert not client.chat.completions.parse.called
    # schema injected into the system message, not the bare legacy prompt
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["response_format"] == {"type": "json_object"}
    assert any("value" in m["content"] for m in sent["messages"] if m["role"] == "system")


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


@pytest.mark.asyncio
async def test_structured_timeout_maps_to_timeout_error():
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(side_effect=asyncio.TimeoutError())
    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="s", response_model=_verdict_model(),
    )
    assert out.error_kind is JudgeError.TIMEOUT
    assert out.payload is None


@pytest.mark.asyncio
async def test_non_schema_badrequest_does_not_fall_back():
    # A 400 unrelated to schema support (e.g. context length) must NOT be swallowed
    # into the json_object fallback; it surfaces as an API_STATUS error and `create`
    # is never called.
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=BadRequestError(
            message="context length exceeded",
            response=MagicMock(status_code=400), body={"error": {"message": "context length exceeded"}},
        )
    )
    client.chat.completions.create = AsyncMock()  # must not be called
    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="s", response_model=_verdict_model(),
    )
    assert out.error_kind is JudgeError.API_STATUS
    assert not client.chat.completions.create.called


@pytest.mark.asyncio
async def test_parsed_none_without_refusal_is_parse_error():
    # No refusal, but the SDK produced no parsed object (truncation / filter): this is
    # a hard PARSE error, not a clean value=None abstain.
    client = MagicMock()
    msg = MagicMock(); msg.parsed = None; msg.refusal = None
    comp = MagicMock(); comp.choices = [MagicMock(message=msg, finish_reason="length")]; comp.usage = None
    client.chat.completions.parse = AsyncMock(return_value=comp)
    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="s", response_model=_verdict_model(),
    )
    assert out.error_kind is JudgeError.PARSE
    assert out.payload is None


@pytest.mark.asyncio
async def test_json_object_fallback_enforces_label_set():
    # On the fallback path an out-of-set categorical label must raise (-> PARSE),
    # not slip through the loose payload model.
    verdict_model = create_model(
        "LabelVerdict", value=(Literal["yes", "no"], ...), explanation=(str, ...)
    )
    client = MagicMock()
    client.chat.completions.parse = AsyncMock(
        side_effect=BadRequestError(
            message="json_schema not supported",
            response=MagicMock(status_code=400), body={"error": {"message": "json_schema unsupported"}},
        )
    )
    create_comp = MagicMock()
    create_comp.choices = [MagicMock(message=MagicMock(content='{"value": "maybe", "explanation": "x"}'))]
    create_comp.usage = None
    client.chat.completions.create = AsyncMock(return_value=create_comp)

    out = await run_judge(
        client=client, model="m", cfg=_cfg(),
        prompt_template="t", replacements={}, system_prompt="sys", response_model=verdict_model,
    )
    assert out.error_kind is JudgeError.PARSE
