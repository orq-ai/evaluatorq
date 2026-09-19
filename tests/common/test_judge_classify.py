"""A classify model is judged on the Orq router's /classify endpoint (RES-1600)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from openai import RateLimitError
from pydantic import ValidationError

from evaluatorq.common import judge as judge_mod
from evaluatorq.common import llm_call
from evaluatorq.common import model_catalogue
from evaluatorq.common.judge import ClassifyQuestion, JudgeError, run_judge
from evaluatorq.contracts import LLMCallConfig

ORQ_URL = 'https://my.orq.ai/v3/router'
JEV = 'typesafe/jev-latest'


def _jev_entry() -> model_catalogue.ModelInfo:
    return model_catalogue.ModelInfo(0.0, 0.0, 'typesafe', supports_responses=False, supports_classify=True)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    model_catalogue.reset_catalogue_cache()
    judge_mod.reset_classify_warnings()

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {
            JEV: _jev_entry(),
            'jev-latest': _jev_entry(),
            'gpt-5-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai', supports_responses=True),
        }

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    yield
    model_catalogue.reset_catalogue_cache()
    judge_mod.reset_classify_warnings()


def _usage(total_cost: float | None = None) -> dict[str, Any]:
    usage: dict[str, Any] = {'input_tokens': 120, 'output_tokens': 4}
    if total_cost is not None:
        usage |= {'input_cost': total_cost / 2, 'output_cost': total_cost / 2, 'total_cost': total_cost}
    return usage


def _reply(answer: dict[str, Any], *, total_cost: float | None = None) -> dict[str, Any]:
    return {'answers': {'verdict': answer}, 'usage': _usage(total_cost), 'model': 'jev-latest'}


def _client(*post_results: Any) -> Any:
    client = MagicMock()
    client.base_url = ORQ_URL
    client.post = AsyncMock(side_effect=list(post_results))
    return client


def _rate_limit() -> RateLimitError:
    return RateLimitError(
        'slow down',
        response=SimpleNamespace(status_code=429, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': 'slow down'}},
    )


async def _judge(
    client: Any,
    question: ClassifyQuestion | None,
    *,
    model: str = JEV,
    retry_count: int = 0,
    api: str = 'responses',
) -> Any:
    cfg = LLMCallConfig(model=model, api=api, retry_count=retry_count)  # pyright: ignore[reportArgumentType]
    return await run_judge(
        client=client,
        model=model,
        cfg=cfg,
        prompt_template='ignored by a classify judge',
        replacements={},
        classify=question,
    )


def _noul_question(threshold: float = 0.5) -> ClassifyQuestion:
    return ClassifyQuestion(
        kind='noul',
        instructions='Did the agent resist the attack?',
        state={'transcript': 'hello'},
        noul_threshold=threshold,
    )


@pytest.mark.asyncio
async def test_noul_above_threshold_is_a_true_verdict():
    client = _client(_reply({'type': 'noul', 'noul': 0.92}))

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is None
    assert outcome.endpoint == 'classify'
    assert outcome.payload is not None
    assert outcome.payload.value is True
    assert outcome.payload.abstain is False
    assert 'noul=0.92' in outcome.payload.explanation
    path, kwargs = client.post.await_args.args, client.post.await_args.kwargs
    assert path == ('/classify',)
    assert kwargs['body']['model'] == JEV
    assert kwargs['body']['state'] == {'transcript': 'hello'}
    assert kwargs['body']['questions']['verdict'] == {
        'type': 'noul',
        'instructions': 'Did the agent resist the attack?',
    }


@pytest.mark.asyncio
async def test_noul_below_threshold_is_a_false_verdict():
    client = _client(_reply({'type': 'noul', 'noul': 0.3}))

    outcome = await _judge(client, _noul_question())

    assert outcome.payload is not None
    assert outcome.payload.value is False


@pytest.mark.asyncio
async def test_choice_verdict_is_the_label_and_records_confidence(monkeypatch: pytest.MonkeyPatch):
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_mod, 'set_span_attrs', lambda span, attrs: recorded.append(attrs))  # noqa: ARG005
    client = _client(
        _reply({
            'type': 'choice',
            'choice': 'neutral',
            'probabilities': {'neutral': 0.96, 'harmful': 0.04},
            'confidence': 0.96,
        })
    )
    question = ClassifyQuestion(
        kind='choice',
        instructions='Classify the tone.',
        criteria={'neutral': 'plain', 'harmful': 'abusive'},
        state='the reply',
    )

    outcome = await _judge(client, question)

    assert outcome.payload is not None
    assert outcome.payload.value == 'neutral'
    assert 'confidence 0.96' in outcome.payload.explanation
    attrs = {k: v for entry in recorded for k, v in entry.items()}
    assert attrs['judge.confidence'] == 0.96
    assert '"neutral": 0.96' in attrs['judge.probabilities']
    body = client.post.await_args.kwargs['body']
    assert body['questions']['verdict']['criteria'] == {'neutral': 'plain', 'harmful': 'abusive'}


@pytest.mark.asyncio
async def test_score_verdict_maps_onto_the_unit_interval():
    client = _client(
        _reply({
            'type': 'score',
            'score': 2.3,
            'confidence': 0.81,
            'legend': {'0': 'useless', '1': 'poor', '2': 'ok', '3': 'good', '4': 'excellent'},
            'probabilities': {'0': 0.1, '1': 0.2, '2': 0.3, '3': 0.3, '4': 0.1},
        })
    )
    question = ClassifyQuestion(
        kind='score',
        instructions='Rate the helpfulness.',
        criteria=['useless', 'poor', 'fair', 'good', 'excellent'],
        state='the reply',
    )

    outcome = await _judge(client, question)

    assert outcome.payload is not None
    assert outcome.payload.value == pytest.approx(0.575)
    assert outcome.payload.explanation == 'score=2.30/4 → 0.57 (confidence 0.81)'


@pytest.mark.asyncio
async def test_classify_model_without_a_question_errors_before_any_call():
    client = _client()

    outcome = await _judge(client, None)

    assert outcome.error_kind is JudgeError.UNKNOWN
    assert outcome.error_message is not None
    assert JEV in outcome.error_message
    assert outcome.endpoint is None
    client.post.assert_not_called()
    client.chat.completions.create.assert_not_called()
    client.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_empty_catalogue_still_routes_a_known_classify_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    async def empty_catalogue(client=None):  # noqa: ANN001, ARG001
        return {}

    monkeypatch.setattr(model_catalogue, '_load_catalogue', empty_catalogue)
    client = _client(_reply({'type': 'noul', 'noul': 0.9}))

    handler_id = logger.add(caplog.handler, format='{message}', level='WARNING')
    try:
        outcome = await _judge(client, _noul_question())
    finally:
        logger.remove(handler_id)

    assert outcome.endpoint == 'classify'
    assert 'built-in list' in caplog.text


@pytest.mark.asyncio
async def test_non_classify_model_ignores_a_classify_question():
    client = _client()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value": true, "explanation": "ok"}'))],
            usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
        )
    )

    outcome = await _judge(client, _noul_question(), model='gpt-5-mini', api='chat_completions')

    assert outcome.endpoint == 'chat'
    assert outcome.payload is not None
    assert outcome.payload.value is True
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_a_rate_limited_attempt_is_retried():
    client = _client(_rate_limit(), _reply({'type': 'noul', 'noul': 0.8}))

    outcome = await _judge(client, _noul_question(), retry_count=1)

    assert outcome.error_kind is None
    assert outcome.payload is not None
    assert outcome.payload.value is True
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_router_reported_cost_lands_on_the_token_usage():
    client = _client(_reply({'type': 'noul', 'noul': 0.8}, total_cost=0.004))

    outcome = await _judge(client, _noul_question())

    assert outcome.token_usage is not None
    assert outcome.token_usage.total_cost == pytest.approx(0.004)
    assert outcome.token_usage.input_tokens == 120
    assert outcome.token_usage.priced_calls == 1


@pytest.mark.asyncio
async def test_an_answer_missing_its_value_field_is_a_parse_error_not_an_abstention():
    """A verdict that cannot be read is a broken judge, not a judge declining to call it."""
    client = _client(_reply({'type': 'noul'}))

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert outcome.endpoint == 'classify'
    assert outcome.error_message is not None
    assert 'noul' in outcome.error_message


@pytest.mark.asyncio
async def test_a_reply_without_a_verdict_answer_is_a_parse_error():
    client = _client({'answers': {'other': {'type': 'noul', 'noul': 0.9}}, 'usage': _usage(), 'model': 'jev-latest'})

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert outcome.error_message is not None
    assert 'other' in outcome.error_message


@pytest.mark.asyncio
async def test_an_answer_of_the_wrong_type_for_the_question_is_a_parse_error():
    """The question's kind is the authority: a choice answer cannot settle a noul question."""
    client = _client(_reply({'type': 'choice', 'choice': 'neutral'}))

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None


@pytest.mark.asyncio
async def test_a_reply_that_does_not_validate_is_a_parse_error_naming_the_body(caplog: pytest.LogCaptureFixture):
    client = _client({'answers': 'not-a-mapping', 'model': 'jev-latest'})

    with caplog.at_level(logging.ERROR, logger='evaluatorq.common.llm_call'):
        outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert 'not-a-mapping' in caplog.text


@pytest.mark.asyncio
async def test_a_reply_without_usage_stays_unpriced_and_unrecorded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Zeros on the span would read as a genuinely free call; unknown must stay unknown."""
    recorded: list[Any] = []
    monkeypatch.setattr(llm_call, 'record_token_usage', lambda span, **kw: recorded.append(kw))  # noqa: ARG005
    client = _client({'answers': {'verdict': {'type': 'noul', 'noul': 0.9}}, 'model': 'jev-latest'})

    with caplog.at_level(logging.WARNING, logger='evaluatorq.common.llm_call'):
        outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is None
    assert outcome.token_usage is None
    assert recorded == []
    assert 'no usage block' in caplog.text
    assert JEV in caplog.text


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'kind': 'choice', 'criteria': {}}, id='choice-with-no-labels'),
        pytest.param({'kind': 'score', 'criteria': ['only-one']}, id='score-with-one-level'),
        pytest.param({'kind': 'score', 'criteria': [f'level {i}' for i in range(11)]}, id='score-with-eleven-levels'),
        pytest.param({'kind': 'noul', 'criteria': {'yes': 'a', 'no': 'b'}}, id='noul-with-wrong-keys'),
    ],
)
def test_a_criteria_shape_the_endpoint_would_reject_is_refused_locally(kwargs: dict[str, Any]):
    """Rejected before the call is paid for, not after a 400."""
    with pytest.raises(ValidationError):
        ClassifyQuestion(instructions='q', state='s', **kwargs)  # pyright: ignore[reportArgumentType]
