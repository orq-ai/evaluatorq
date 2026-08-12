"""Judges default to the Orq router's priced Responses endpoint (RES-1295)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import BadRequestError
from pydantic import BaseModel

from evaluatorq.common import judge as judge_mod
from evaluatorq.common import model_catalogue
from evaluatorq.common.judge import run_judge
from evaluatorq.contracts import LLMCallConfig


class Verdict(BaseModel):
    value: bool
    explanation: str


ORQ_URL = 'https://my.orq.ai/v3/router'


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    judge_mod.reset_responses_rejectors()

    async def fake_load():
        return {'gpt-5-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai')}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    yield
    judge_mod.reset_responses_rejectors()


def _responses_reply() -> Any:
    return SimpleNamespace(
        output_parsed=Verdict(value=True, explanation='resisted'),
        output_text='{"value": true, "explanation": "resisted"}',
        usage={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15, 'total_cost': 0.25},
    )


def _chat_reply() -> Any:
    message = SimpleNamespace(parsed=Verdict(value=True, explanation='resisted'), refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason='stop')],
        usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
    )


class _Client:
    """Minimal stand-in recording which endpoint the judge reached for."""

    def __init__(self, base_url: str = ORQ_URL, *, responses_error: Exception | None = None):
        self.base_url = base_url
        self.calls: list[str] = []
        self.models: list[str] = []
        self._responses_error = responses_error
        client = self

        async def responses_parse(**kwargs: Any) -> Any:
            client.calls.append('responses')
            client.models.append(kwargs['model'])
            if client._responses_error is not None:
                raise client._responses_error
            return _responses_reply()

        async def chat_parse(**kwargs: Any) -> Any:
            client.calls.append('chat')
            client.models.append(kwargs['model'])
            return _chat_reply()

        self.responses = SimpleNamespace(parse=responses_parse)
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=chat_parse))


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(
        message,
        response=SimpleNamespace(status_code=400, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': message}},
    )


async def _judge(client: Any, api: str = 'responses') -> Any:
    cfg = LLMCallConfig(model='gpt-5-mini', api=api, max_tokens=256)  # pyright: ignore[reportArgumentType]
    return await run_judge(
        client=client,
        model='gpt-5-mini',
        cfg=cfg,
        prompt_template='judge this',
        replacements={},
        response_model=Verdict,
    )


@pytest.mark.asyncio
async def test_orq_client_uses_responses_and_qualifies_the_model():
    client = _Client()
    outcome = await _judge(client)
    assert client.calls == ['responses']
    assert client.models == ['openai/gpt-5-mini']
    assert outcome.payload is not None and outcome.payload.value is True
    assert outcome.token_usage is not None and outcome.token_usage.total_cost == 0.25


@pytest.mark.asyncio
async def test_non_orq_client_stays_on_chat_completions():
    client = _Client(base_url='https://api.openai.com/v1')
    await _judge(client)
    assert client.calls == ['chat']


@pytest.mark.asyncio
async def test_chat_completions_opt_out_is_honoured():
    client = _Client()
    await _judge(client, api='chat_completions')
    assert client.calls == ['chat']


@pytest.mark.asyncio
async def test_unknown_model_cannot_be_qualified_so_stays_on_chat(monkeypatch: pytest.MonkeyPatch):
    async def empty_catalogue():
        return {}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', empty_catalogue)
    client = _Client()
    await _judge(client)
    assert client.calls == ['chat']


@pytest.mark.asyncio
async def test_rejected_responses_endpoint_falls_back_and_is_remembered():
    client = _Client(responses_error=_bad_request('endpoint not supported for this model'))
    first = await _judge(client)
    assert client.calls == ['responses', 'chat']
    assert first.payload is not None and first.payload.value is True

    await _judge(client)
    # Second judgement skips the endpoint that already 400'd.
    assert client.calls == ['responses', 'chat', 'chat']
