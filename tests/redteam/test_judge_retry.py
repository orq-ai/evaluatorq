"""Retry-path coverage for ``run_judge`` (RES-1295 follow-up).

Every other judge test opts out of retry with ``retry_count=0``, so the
``with_retry`` wrapper around ``_attempt`` in ``src/evaluatorq/common/judge.py``
had zero coverage before this file. These tests exercise: a retryable failure
that recovers, retries exhausted landing in the normal error classification, a
non-retryable failure consuming exactly one attempt (the expensive regression —
turning every malformed verdict into N paid calls), the ``retry_count=0``
opt-out, and the new ``JudgeOutcome.endpoint`` field across the Responses/chat/
fallback paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIStatusError, BadRequestError, RateLimitError
from pydantic import BaseModel

from evaluatorq.common import judge as judge_mod
from evaluatorq.common import model_catalogue
from evaluatorq.common.judge import JudgeError, run_judge
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.redteam.contracts import EvaluatorConfig

ORQ_URL = 'https://my.orq.ai/v3/router'
OPENAI_URL = 'https://api.openai.com/v1'


class Verdict(BaseModel):
    value: bool
    explanation: str


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    judge_mod.reset_responses_rejectors()

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {'gpt-5-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai', supports_responses=True)}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    yield
    judge_mod.reset_responses_rejectors()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch):
    """Retry backoff defaults to RETRY_MIN_WAIT_S=2.0s; don't actually wait."""

    async def fast_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(judge_mod.asyncio, 'sleep', fast_sleep)


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


def _rate_limit_error() -> RateLimitError:
    return RateLimitError(
        'rate limited',
        response=SimpleNamespace(status_code=429, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': 'rate limited'}},
    )


def _server_error(status: int = 503) -> APIStatusError:
    return APIStatusError(
        'server error',
        response=SimpleNamespace(status_code=status, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': 'server error'}},
    )


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(
        message,
        response=SimpleNamespace(status_code=400, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': message}},
    )


class _Client:
    """Minimal stand-in whose ``responses.parse``/``chat.completions.*`` are swappable."""

    def __init__(self, base_url: str = ORQ_URL):
        self.base_url = base_url
        self.calls: list[str] = []
        self.responses = SimpleNamespace(parse=self._default_responses_parse)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=self._default_chat_parse, create=self._default_chat_create)
        )

    async def _default_responses_parse(self, **kwargs: Any) -> Any:
        self.calls.append('responses')
        return _responses_reply()

    async def _default_chat_parse(self, **kwargs: Any) -> Any:
        self.calls.append('chat')
        return _chat_reply()

    async def _default_chat_create(self, **kwargs: Any) -> Any:
        self.calls.append('chat')
        reply = _chat_reply()
        reply.choices[0].message = SimpleNamespace(content='{"value": true, "explanation": "resisted"}')
        return reply


async def _judge(
    client: Any,
    *,
    api: str = 'responses',
    response_model: type[BaseModel] | None = Verdict,
    structured_output: bool = True,
    retry_count: int = 2,
) -> Any:
    cfg = LLMCallConfig(
        model='gpt-5-mini',
        api=api,  # pyright: ignore[reportArgumentType]
        max_tokens=256,
        retry_count=retry_count,
    )
    return await run_judge(
        client=client,
        model='gpt-5-mini',
        cfg=cfg,
        prompt_template='judge this',
        replacements={},
        response_model=response_model,
        structured_output=structured_output,
    )


# --- 1. Retryable failure recovers on attempt 2 --------------------------------


@pytest.mark.asyncio
async def test_retryable_rate_limit_recovers_on_second_attempt():
    client = _Client()
    attempts = {'n': 0}

    async def flaky_responses_parse(**kwargs: Any) -> Any:
        attempts['n'] += 1
        client.calls.append('responses')
        if attempts['n'] == 1:
            raise _rate_limit_error()
        return _responses_reply()

    client.responses = SimpleNamespace(parse=flaky_responses_parse)
    outcome = await _judge(client)

    assert attempts['n'] == 2
    assert outcome.payload is not None and outcome.payload.value is True
    assert outcome.error_kind is None
    assert outcome.endpoint == 'responses'


@pytest.mark.asyncio
async def test_retryable_503_on_chat_recovers_on_second_attempt():
    client = _Client(base_url=OPENAI_URL)  # non-Orq -> stays on chat
    attempts = {'n': 0}

    async def flaky_chat_parse(**kwargs: Any) -> Any:
        attempts['n'] += 1
        client.calls.append('chat')
        if attempts['n'] == 1:
            raise _server_error(503)
        return _chat_reply()

    client.chat = SimpleNamespace(completions=SimpleNamespace(parse=flaky_chat_parse))
    outcome = await _judge(client)

    assert attempts['n'] == 2
    assert outcome.payload is not None and outcome.payload.value is True
    assert outcome.endpoint == 'chat'


# --- 0. Default retry_count -----------------------------------------------------


def test_llm_call_config_default_retry_count_is_one():
    assert LLMCallConfig().retry_count == 1


def test_evaluator_config_default_retry_count_is_one():
    assert EvaluatorConfig().retry_count == 1


@pytest.mark.asyncio
async def test_default_retry_count_issues_exactly_two_requests():
    """One retry after the initial call: a permanently-failing judge with the
    default ``LLMCallConfig`` issues exactly 2 requests, not 3 (the old
    total-attempts default) and not 1 (no retry at all)."""
    client = _Client()

    async def always_fails(**kwargs: Any) -> Any:
        client.calls.append('responses')
        raise _server_error(503)

    client.responses = SimpleNamespace(parse=always_fails)
    cfg = LLMCallConfig(model='gpt-5-mini', api='responses', max_tokens=256)
    assert cfg.retry_count == 1
    outcome = await run_judge(
        client=client,  # pyright: ignore[reportArgumentType]
        model='gpt-5-mini',
        cfg=cfg,
        prompt_template='judge this',
        replacements={},
        response_model=Verdict,
        structured_output=True,
    )

    assert len(client.calls) == 2
    assert outcome.error_kind is JudgeError.API_STATUS


# --- 2. Retries exhausted -> normal error classification, not an unhandled raise --


@pytest.mark.asyncio
async def test_exhausted_retries_classify_as_api_status():
    client = _Client()

    async def always_fails(**kwargs: Any) -> Any:
        client.calls.append('responses')
        raise _server_error(503)

    client.responses = SimpleNamespace(parse=always_fails)
    outcome = await _judge(client, retry_count=2)

    assert len(client.calls) == 3
    assert outcome.error_kind is JudgeError.API_STATUS
    assert outcome.payload is None


# --- 3. Non-retryable failures consume exactly ONE attempt ---------------------


@pytest.mark.asyncio
async def test_malformed_json_consumes_exactly_one_attempt():
    """A ValidationError from unparseable JSON must not be retried (RES-1295 regression)."""
    client = _Client()

    async def malformed_responses_parse(**kwargs: Any) -> Any:
        client.calls.append('responses')
        reply = _responses_reply()
        reply.output_parsed = None
        reply.output_text = 'not json at all {'
        return reply

    client.responses = SimpleNamespace(parse=malformed_responses_parse)
    outcome = await _judge(client, retry_count=2)

    # No parsed object surfaces as JudgeError.PARSE directly from _responses_judge,
    # not a raised ValidationError, but it must still not retry.
    assert len(client.calls) == 1
    assert outcome.error_kind is JudgeError.PARSE


@pytest.mark.asyncio
async def test_validation_error_from_malformed_chat_json_consumes_one_attempt():
    """The legacy json_object path raises ValidationError on unparseable JSON; must not retry."""
    client = _Client(base_url=OPENAI_URL)

    async def malformed_chat_create(**kwargs: Any) -> Any:
        client.calls.append('chat')
        reply = _chat_reply()
        reply.choices[0].message = SimpleNamespace(content='not valid json')
        return reply

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=malformed_chat_create))
    outcome = await _judge(client, retry_count=2, response_model=None)

    assert len(client.calls) == 1
    assert outcome.error_kind is JudgeError.PARSE


@pytest.mark.asyncio
async def test_bad_request_on_chat_path_consumes_one_attempt():
    """A BadRequestError unrelated to schema/response_format must raise straight through, once."""
    client = _Client(base_url=OPENAI_URL)

    async def rejecting_chat_parse(**kwargs: Any) -> Any:
        client.calls.append('chat')
        raise _bad_request('context_length_exceeded')

    client.chat = SimpleNamespace(completions=SimpleNamespace(parse=rejecting_chat_parse))
    outcome = await _judge(client, retry_count=2)

    assert len(client.calls) == 1
    assert outcome.error_kind is JudgeError.API_STATUS


# --- 4. retry_count=0 disables retry -----------------------------------------


@pytest.mark.asyncio
async def test_retry_count_zero_disables_retry():
    client = _Client()

    async def always_fails(**kwargs: Any) -> Any:
        client.calls.append('responses')
        raise _rate_limit_error()

    client.responses = SimpleNamespace(parse=always_fails)
    outcome = await _judge(client, retry_count=0)

    assert len(client.calls) == 1
    assert outcome.error_kind is JudgeError.API_STATUS


# --- JudgeOutcome.endpoint ------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_is_responses_when_served_by_responses_api():
    client = _Client()
    outcome = await _judge(client)
    assert client.calls == ['responses']
    assert outcome.endpoint == 'responses'


@pytest.mark.asyncio
async def test_endpoint_is_chat_when_served_by_chat_completions():
    client = _Client(base_url=OPENAI_URL)
    outcome = await _judge(client)
    assert client.calls == ['chat']
    assert outcome.endpoint == 'chat'


@pytest.mark.asyncio
async def test_endpoint_is_chat_when_responses_400s_and_falls_back():
    async def rejecting_responses_parse(**kwargs: Any) -> Any:
        raise _bad_request('endpoint not supported for this model')

    client = _Client()
    client.responses = SimpleNamespace(parse=rejecting_responses_parse)
    outcome = await _judge(client)

    assert outcome.endpoint == 'chat'
    assert outcome.payload is not None and outcome.payload.value is True


# --- JudgeOutcome.endpoint on the terminal error returns ------------------------
#
# A panel aggregates endpoints over decisive AND failed predictions
# (`common.jury._run_jury_core`), so an unstamped failure makes a whole panel's
# provenance read `None` — "nothing recorded an endpoint" rather than "the
# Responses call timed out". These pin which endpoint each terminal return names.


@pytest.mark.asyncio
async def test_timeout_on_responses_stamps_responses_endpoint():
    client = _Client()

    async def times_out(**kwargs: Any) -> Any:
        client.calls.append('responses')
        raise TimeoutError

    client.responses = SimpleNamespace(parse=times_out)
    outcome = await _judge(client, retry_count=0)

    assert outcome.error_kind is JudgeError.TIMEOUT
    assert outcome.endpoint == 'responses'


@pytest.mark.asyncio
async def test_timeout_on_chat_stamps_chat_endpoint():
    client = _Client(base_url=OPENAI_URL)  # non-Orq -> never leaves chat

    async def times_out(**kwargs: Any) -> Any:
        client.calls.append('chat')
        raise TimeoutError

    client.chat = SimpleNamespace(completions=SimpleNamespace(parse=times_out))
    outcome = await _judge(client, retry_count=0)

    assert outcome.error_kind is JudgeError.TIMEOUT
    assert outcome.endpoint == 'chat'


@pytest.mark.asyncio
async def test_exhausted_retries_stamp_the_responses_endpoint():
    client = _Client()

    async def always_fails(**kwargs: Any) -> Any:
        client.calls.append('responses')
        raise _server_error(503)

    client.responses = SimpleNamespace(parse=always_fails)
    outcome = await _judge(client, retry_count=1)

    assert outcome.error_kind is JudgeError.API_STATUS
    assert outcome.endpoint == 'responses'


@pytest.mark.asyncio
async def test_parse_error_on_chat_stamps_chat_endpoint():
    """The legacy json_object path's unparseable verdict: a ValidationError still
    knows it was chat that served it."""
    client = _Client(base_url=OPENAI_URL)

    async def malformed_chat_create(**kwargs: Any) -> Any:
        client.calls.append('chat')
        reply = _chat_reply()
        reply.choices[0].message = SimpleNamespace(content='not valid json')
        return reply

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=malformed_chat_create))
    outcome = await _judge(client, retry_count=0, response_model=None)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.endpoint == 'chat'


@pytest.mark.asyncio
async def test_failure_after_responses_fallback_stamps_chat_not_responses():
    """The regression a naive `cfg.api` read would cause: cfg says 'responses',
    but the Responses call 400'd and it was the chat retry that actually failed."""
    client = _Client()

    async def rejecting_responses_parse(**kwargs: Any) -> Any:
        client.calls.append('responses')
        raise _bad_request('responses endpoint not supported for this model')

    async def failing_chat_parse(**kwargs: Any) -> Any:
        client.calls.append('chat')
        raise _server_error(503)

    client.responses = SimpleNamespace(parse=rejecting_responses_parse)
    client.chat = SimpleNamespace(completions=SimpleNamespace(parse=failing_chat_parse))
    outcome = await _judge(client, retry_count=0)

    assert client.calls == ['responses', 'chat']
    assert outcome.error_kind is JudgeError.API_STATUS
    assert outcome.endpoint == 'chat'


@pytest.mark.asyncio
async def test_unknown_failure_leaves_endpoint_unset():
    """The catch-all also covers failures that never reached a provider, so it
    names no endpoint: None is the honest answer, not a guess."""
    client = _Client()

    async def explodes(**kwargs: Any) -> Any:
        raise RuntimeError('something entirely else')

    client.responses = SimpleNamespace(parse=explodes)
    outcome = await _judge(client, retry_count=0)

    assert outcome.error_kind is JudgeError.UNKNOWN
    assert outcome.endpoint is None
