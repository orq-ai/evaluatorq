"""Judges default to the Orq router's priced Responses endpoint (RES-1295)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, BadRequestError
from pydantic import BaseModel

from evaluatorq.common import judge as judge_mod
from evaluatorq.common import model_catalogue
from evaluatorq.common.judge import JudgeError, run_judge
from evaluatorq.common.retry import without_client_retries
from evaluatorq.contracts import LLMCallConfig


class Verdict(BaseModel):
    value: bool
    explanation: str


class AbstainingVerdict(BaseModel):
    value: bool
    explanation: str
    abstain: bool


ORQ_URL = 'https://my.orq.ai/v3/router'


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    judge_mod.reset_responses_rejectors()

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {'gpt-5-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai', supports_responses=True)}

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

        async def chat_create(**kwargs: Any) -> Any:
            client.calls.append('chat')
            client.models.append(kwargs['model'])
            reply = _chat_reply()
            reply.choices[0].message = SimpleNamespace(content='{"value": true, "explanation": "resisted"}')
            return reply

        self.responses = SimpleNamespace(parse=responses_parse)
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=chat_parse, create=chat_create))


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(
        message,
        response=SimpleNamespace(status_code=400, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': message}},
    )


async def _judge(
    client: Any,
    api: str = 'responses',
    *,
    response_model: type[BaseModel] | None = Verdict,
    structured_output: bool = True,
) -> Any:
    cfg = LLMCallConfig(model='gpt-5-mini', api=api, max_tokens=256)  # pyright: ignore[reportArgumentType]
    return await run_judge(
        client=client,
        model='gpt-5-mini',
        cfg=cfg,
        prompt_template='judge this',
        replacements={},
        response_model=response_model,
        structured_output=structured_output,
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
async def test_responses_payload_rebuild_preserves_abstain(monkeypatch: pytest.MonkeyPatch):
    parsed = AbstainingVerdict(value=False, explanation='uncertain', abstain=True)

    async def fake_execute_response(**_kwargs: Any) -> tuple[Any, Any]:
        return SimpleNamespace(output_parsed=parsed, output_text=parsed.model_dump_json()), None

    monkeypatch.setattr(judge_mod, 'execute_response', fake_execute_response)
    outcome = await judge_mod._responses_judge(
        client=MagicMock(),
        model='m',
        cfg=LLMCallConfig(),
        system_prompt='sys',
        user_prompt='user',
        span=None,
        temp=None,
        response_model=AbstainingVerdict,
    )

    assert outcome.payload is not None
    assert outcome.payload.value is False
    assert outcome.payload.abstain is True


@pytest.mark.asyncio
async def test_without_a_response_model_the_verdict_schema_is_enforced():
    """No caller model — the payload schema is sent rather than a bare json_object."""
    client = _Client()
    seen: dict[str, Any] = {}

    async def responses_parse(**kwargs: Any) -> Any:
        seen.update(kwargs)
        client.calls.append('responses')
        return _responses_reply()

    client.responses = SimpleNamespace(parse=responses_parse)
    await _judge(client, response_model=None)
    assert client.calls == ['responses']
    assert seen['text_format'] is judge_mod.EvaluatorResponsePayload
    assert 'text' not in seen


@pytest.mark.asyncio
async def test_structured_output_opt_out_stays_on_chat_completions():
    """The Responses path is schema-only, so a caller that cannot do schemas skips it."""
    client = _Client()
    await _judge(client, structured_output=False)
    assert client.calls == ['chat']


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
    async def empty_catalogue(client=None):  # noqa: ANN001, ARG001
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


@pytest.mark.asyncio
async def test_unparsed_response_keeps_its_usage():
    """A billed call that produced no verdict still reports its tokens and cost."""
    client = _Client()

    async def responses_parse(**_kwargs: Any) -> Any:
        client.calls.append('responses')
        reply = _responses_reply()
        reply.output_parsed = None
        return reply

    client.responses = SimpleNamespace(parse=responses_parse)
    outcome = await _judge(client)
    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.token_usage is not None and outcome.token_usage.total_cost == 0.25
    assert outcome.raw_content == '{"value": true, "explanation": "resisted"}'


@pytest.mark.asyncio
async def test_unrelated_bad_request_does_not_downgrade_the_model():
    """A content-policy 400 is not evidence the endpoint is unavailable."""
    client = _Client(responses_error=_bad_request('content management policy violation'))
    await _judge(client)
    assert client.calls == ['responses', 'chat']

    await _judge(client)
    # Still tries the priced endpoint: only an endpoint/param 400 is remembered.
    assert client.calls == ['responses', 'chat', 'responses', 'chat']


@pytest.mark.asyncio
async def test_a_retry_does_not_re_pay_the_rejected_endpoint():
    """The chat fallback failing retryably must not send the 400 again."""
    attempts = {'n': 0}
    client = _Client(responses_error=_bad_request('endpoint not supported for this model'))

    async def chat_parse(**kwargs: Any) -> Any:
        client.calls.append('chat')
        client.models.append(kwargs['model'])
        attempts['n'] += 1
        if attempts['n'] == 1:
            raise APIConnectionError(request=None)  # pyright: ignore[reportArgumentType]
        return _chat_reply()

    client.chat = SimpleNamespace(completions=SimpleNamespace(parse=chat_parse))
    outcome = await _judge(client)
    assert outcome.payload is not None and outcome.payload.value is True
    assert client.calls == ['responses', 'chat', 'chat']


# ---------------------------------------------------------------------------
# without_client_retries: the only thing preventing with_retry x SDK-retry
# multiplication when the caller owns retry.
# ---------------------------------------------------------------------------


class _RetryStubClient:
    """Minimal stand-in exposing only what ``without_client_retries`` reads."""

    def __init__(self, max_retries: int):
        self.max_retries = max_retries
        self.with_options_calls: list[dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> _RetryStubClient:
        self.with_options_calls.append(kwargs)
        disarmed = _RetryStubClient(kwargs.get('max_retries', self.max_retries))
        disarmed.with_options_calls = self.with_options_calls
        return disarmed


def test_disarms_client_when_caller_owns_retry_and_client_has_a_budget():
    """``run_judge`` retries the whole attempt via ``with_retry``; a client with
    its own nonzero SDK retry budget must be disarmed or the two multiply."""
    client = _RetryStubClient(max_retries=3)

    result = without_client_retries(cast('Any', client))

    assert result is not client
    assert client.with_options_calls == [{'max_retries': 0}]
    assert result.max_retries == 0


def test_client_is_disarmed_even_when_the_outer_retry_budget_is_one_attempt():
    """The retry boundary owns the call even when its configured budget is one attempt."""
    client = _RetryStubClient(max_retries=3)

    result = without_client_retries(cast('Any', client))

    assert result is not client
    assert client.with_options_calls == [{'max_retries': 0}]
    assert result.max_retries == 0


def test_client_passed_through_untouched_when_it_has_no_retry_budget():
    """A client already built with ``max_retries=0`` has nothing to disarm —
    calling ``with_options`` would be a needless extra client object."""
    client = _RetryStubClient(max_retries=0)

    result = without_client_retries(cast('Any', client))

    assert result is client
    assert client.with_options_calls == []
