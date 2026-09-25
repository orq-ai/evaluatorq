"""A classify model is judged on the Orq router's /classify endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from openai import APITimeoutError, RateLimitError
from pydantic import ValidationError

import evaluatorq
from evaluatorq.common import judge as judge_mod
from evaluatorq.common import llm_call
from evaluatorq.common import model_catalogue
from evaluatorq.common import tracing
from evaluatorq.common.judge import ClassifyOutcome, ClassifyQuestion, ClassifyRequest, JudgeError, run_classify, run_judge
from evaluatorq.contracts import LLMCallConfig

ORQ_URL = 'https://my.orq.ai/v3/router'
JEV = 'typesafe/jev-latest'


def test_classify_question_is_a_public_contract() -> None:
    assert evaluatorq.ClassifyQuestion is ClassifyQuestion
    assert evaluatorq.ClassifyRequest is ClassifyRequest
    assert evaluatorq.ClassifyOutcome is ClassifyOutcome
    assert evaluatorq.run_classify is run_classify


def test_classify_request_requires_a_question() -> None:
    with pytest.raises(ValidationError):
        ClassifyRequest(state='reply', questions={})


def _jev_entry() -> model_catalogue.ModelInfo:
    return model_catalogue.ModelInfo(0.0, 0.0, 'typesafe', supports_responses=False, supports_classify=True)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    model_catalogue.reset_catalogue_cache()

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {
            JEV: _jev_entry(),
            'jev-latest': _jev_entry(),
            'gpt-5-mini': model_catalogue.ModelInfo(0.00025, 0.002, 'openai', supports_responses=True),
        }

    monkeypatch.setattr(model_catalogue, '_load_catalogue', fake_load)
    yield
    model_catalogue.reset_catalogue_cache()


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


@pytest.mark.parametrize('threshold', [-0.1, 1.1, float('nan'), float('inf')])
def test_noul_question_rejects_an_invalid_probability_threshold(threshold: float) -> None:
    with pytest.raises(ValidationError):
        _noul_question(threshold)


@pytest.mark.asyncio
async def test_noul_above_threshold_is_a_true_verdict():
    client = _client(_reply({'type': 'noul', 'noul': 0.92}))

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is None
    assert outcome.endpoint == 'classify'
    assert outcome.payload is not None
    assert outcome.payload.value is True
    assert outcome.payload.abstain is False
    assert outcome.payload.explanation == 'noul=0.92 (threshold 0.5)'
    path, kwargs = client.post.await_args.args, client.post.await_args.kwargs
    assert path == ('/classify',)
    assert kwargs['body']['model'] == JEV
    assert kwargs['body']['state'] == {'transcript': 'hello'}
    assert kwargs['body']['questions']['verdict'] == {
        'type': 'noul',
        'instructions': 'Did the agent resist the attack?',
    }


@pytest.mark.asyncio
async def test_run_classify_answers_multiple_questions_in_one_request() -> None:
    client = _client(
        {
            'answers': {
                'tone': {'type': 'choice', 'choice': 'neutral'},
                'risk': {'type': 'noul', 'noul': 0.2},
            },
            'usage': _usage(),
            'model': 'jev-latest',
        }
    )
    request = ClassifyRequest(
        state={'reply': 'hello'},
        questions={
            'tone': ClassifyQuestion(
                kind='choice',
                instructions='Classify the tone.',
                criteria={'neutral': 'plain'},
                state='ignored question state',
            ),
            'risk': ClassifyQuestion(
                kind='noul',
                instructions='Is the reply risky?',
                state={'ignored': True},
            ),
        },
    )

    outcome = await run_classify(client=client, model=JEV, cfg=LLMCallConfig(model=JEV), request=request)

    assert outcome.error_kind is None
    assert outcome.response is not None
    assert set(outcome.response.answers) == {'tone', 'risk'}
    assert outcome.token_usage is not None
    assert outcome.token_usage.input_tokens == 120
    body = client.post.await_args.kwargs['body']
    assert body == {
        'model': JEV,
        'state': {'reply': 'hello'},
        'questions': {
            'tone': {'type': 'choice', 'instructions': 'Classify the tone.', 'criteria': {'neutral': 'plain'}},
            'risk': {'type': 'noul', 'instructions': 'Is the reply risky?'},
        },
    }


@pytest.mark.asyncio
async def test_run_classify_disables_an_injected_clients_sdk_retries() -> None:
    client = _client()
    client.max_retries = 2
    retryless = _client(_reply({'type': 'noul', 'noul': 0.9}))
    retryless.max_retries = 0
    client.with_options.return_value = retryless

    outcome = await run_classify(
        client=client,
        model=JEV,
        cfg=LLMCallConfig(model=JEV),
        request=ClassifyRequest(state='reply', questions={'verdict': _noul_question()}),
    )

    assert outcome.error_kind is None
    client.with_options.assert_called_once_with(max_retries=0)
    client.post.assert_not_awaited()
    retryless.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_classify_reports_requested_answer_keys_that_are_missing() -> None:
    client = _client(
        {
            'answers': {'tone': {'type': 'choice', 'choice': 'neutral'}},
            'usage': _usage(),
            'model': 'jev-latest',
        }
    )
    request = ClassifyRequest(
        state='reply',
        questions={
            'tone': ClassifyQuestion(kind='choice', instructions='Classify it.', criteria={'neutral': 'plain'}, state='x'),
            'risk': ClassifyQuestion(kind='noul', instructions='Is it risky?', state='x'),
        },
    )

    outcome = await run_classify(client=client, model=JEV, cfg=LLMCallConfig(model=JEV), request=request)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.response is None
    assert outcome.error_message is not None
    assert 'risk' in outcome.error_message


@pytest.mark.asyncio
async def test_run_classify_keeps_usage_when_an_answer_fails_validation() -> None:
    client = _client(_reply({'type': 'noul', 'noul': 'invalid'}))
    request = ClassifyRequest(state='reply', questions={'verdict': _noul_question()})

    outcome = await run_classify(client=client, model=JEV, cfg=LLMCallConfig(model=JEV), request=request)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.token_usage is not None
    assert outcome.token_usage.input_tokens == 120
    assert outcome.token_usage.output_tokens == 4


@pytest.mark.asyncio
async def test_run_classify_maps_a_timeout_to_a_timeout_error() -> None:
    client = _client(asyncio.TimeoutError())
    request = ClassifyRequest(state='reply', questions={'risk': _noul_question()})

    outcome = await run_classify(client=client, model=JEV, cfg=LLMCallConfig(model=JEV), request=request)

    assert outcome.error_kind is JudgeError.TIMEOUT


@pytest.mark.asyncio
async def test_run_classify_maps_provider_timeout_to_a_timeout_error() -> None:
    client = _client(APITimeoutError(request=MagicMock()))
    request = ClassifyRequest(state='reply', questions={'risk': _noul_question()})

    outcome = await run_classify(client=client, model=JEV, cfg=LLMCallConfig(model=JEV), request=request)

    assert outcome.error_kind is JudgeError.TIMEOUT


@pytest.mark.asyncio
async def test_classify_judge_reraises_provider_timeout_for_run_judge_retry() -> None:
    timeout = APITimeoutError(request=MagicMock())
    client = _client(timeout)

    with pytest.raises(APITimeoutError) as caught:
        await judge_mod._classify_judge(
            client=client,
            model=JEV,
            cfg=LLMCallConfig(model=JEV),
            question=_noul_question(),
        )

    assert caught.value is timeout


@pytest.mark.asyncio
async def test_noul_below_threshold_is_a_false_verdict():
    client = _client(_reply({'type': 'noul', 'noul': 0.3}))

    outcome = await _judge(client, _noul_question())

    assert outcome.payload is not None
    assert outcome.payload.value is False


@pytest.mark.parametrize(
    ('noul', 'rendered'),
    [
        pytest.param(0.499, '0.499', id='three-decimals'),
        pytest.param(0.499999999, '0.499999999', id='nine-decimals'),
    ],
)
@pytest.mark.asyncio
async def test_noul_explanation_keeps_enough_precision_to_show_which_side_of_threshold_won(
    noul: float,
    rendered: str,
):
    client = _client(_reply({'type': 'noul', 'noul': noul}))

    outcome = await _judge(client, _noul_question())

    assert outcome.payload is not None
    assert outcome.payload.value is False
    assert outcome.payload.explanation == f'noul={rendered} (threshold 0.5)'


@pytest.mark.asyncio
async def test_noul_explanation_formats_a_close_custom_threshold_at_matching_precision() -> None:
    client = _client(_reply({'type': 'noul', 'noul': 0.50000005}))

    outcome = await _judge(client, _noul_question(0.5000001))

    assert outcome.payload is not None
    assert outcome.payload.value is False
    assert outcome.payload.explanation == 'noul=0.50000005 (threshold 0.5000001)'


@pytest.mark.asyncio
async def test_choice_verdict_is_the_label_and_records_confidence(monkeypatch: pytest.MonkeyPatch):
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_mod, 'set_span_attrs', lambda span, attrs: recorded.append(attrs))  # noqa: ARG005
    client = _client(
        _reply({
            'type': 'choice',
            'choice': 'frustrated',
            'probabilities': {'frustrated': 0.86, 'neutral': 0.14},
            'confidence': 0.86,
        })
    )
    question = ClassifyQuestion(
        kind='choice',
        instructions='Classify the tone.',
        criteria={'frustrated': 'upset', 'neutral': 'plain'},
        state='the reply',
    )

    outcome = await _judge(client, question)

    assert outcome.payload is not None
    assert outcome.payload.value == 'frustrated'
    assert 'confidence 0.86' in outcome.payload.explanation
    assert outcome.raw_output == {
        'type': 'choice',
        'choice': 'frustrated',
        'probabilities': {'frustrated': 0.86, 'neutral': 0.14},
        'confidence': 0.86,
    }
    attrs = {k: v for entry in recorded for k, v in entry.items()}
    assert attrs['judge.confidence'] == 0.86
    assert '"frustrated": 0.86' in attrs['judge.probabilities']
    body = client.post.await_args.kwargs['body']
    assert body['questions']['verdict']['criteria'] == {'frustrated': 'upset', 'neutral': 'plain'}


@pytest.mark.asyncio
async def test_classify_raw_output_excludes_absent_answer_fields() -> None:
    client = _client(_reply({'type': 'choice', 'choice': 'neutral'}))
    question = ClassifyQuestion(
        kind='choice',
        instructions='Classify the tone.',
        criteria={'neutral': 'plain', 'harmful': 'abusive'},
        state='the reply',
    )

    outcome = await _judge(client, question)

    assert outcome.raw_output == {'type': 'choice', 'choice': 'neutral'}


@pytest.mark.asyncio
async def test_an_empty_probability_distribution_is_still_recorded(monkeypatch: pytest.MonkeyPatch):
    """An empty distribution is something the model reported; dropping it reads as never reported."""
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_mod, 'set_span_attrs', lambda span, attrs: recorded.append(attrs))  # noqa: ARG005
    client = _client(_reply({'type': 'noul', 'noul': 0.9, 'probabilities': {}}))

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is None
    attrs = {k: v for entry in recorded for k, v in entry.items()}
    assert attrs['judge.probabilities'] == '{}'


@pytest.mark.asyncio
async def test_the_span_input_carries_the_question_as_well_as_the_state(monkeypatch: pytest.MonkeyPatch):
    """A trace reader who sees only the state cannot tell what Jev was asked."""
    recorded: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(llm_call, 'record_llm_input', lambda span, messages: recorded.append(messages))  # noqa: ARG005
    question = ClassifyQuestion(
        kind='choice',
        instructions='Classify the tone.',
        criteria={'neutral': 'plain', 'harmful': 'abusive'},
        state='the reply',
    )
    client = _client(_reply({'type': 'choice', 'choice': 'neutral'}))

    await _judge(client, question)

    assert len(recorded) == 1
    system, user = recorded[0]
    assert system['role'] == 'system'
    assert json.loads(system['content']) == {
        'instructions': 'Classify the tone.',
        'criteria': {'neutral': 'plain', 'harmful': 'abusive'},
    }
    assert user['role'] == 'user'
    assert json.loads(user['content']) == 'the reply'


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
async def test_score_explanation_keeps_precision_near_the_halfway_boundary():
    client = _client(_reply({'type': 'score', 'score': 1.999}))
    question = ClassifyQuestion(
        kind='score',
        instructions='Rate the helpfulness.',
        criteria=['useless', 'poor', 'fair', 'good', 'excellent'],
        state='the reply',
    )

    outcome = await _judge(client, question)

    assert outcome.payload is not None
    assert outcome.payload.value == pytest.approx(0.49975)
    assert outcome.payload.explanation == 'score=1.999/4 → 0.4998'


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
    assert 'neutral' in outcome.raw_content


@pytest.mark.asyncio
async def test_a_choice_outside_the_question_labels_is_a_parse_error():
    """A label nobody offered is not a verdict about this question."""
    question = ClassifyQuestion(
        kind='choice',
        instructions='which is better?',
        criteria={'A': 'A is better', 'B': 'B is better'},
        state={'pair': 'x'},
    )
    client = _client(_reply({'type': 'choice', 'choice': 'C'}))

    outcome = await _judge(client, question)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert "'choice': 'C'" in outcome.raw_content or '"choice":"C"' in outcome.raw_content


@pytest.mark.asyncio
async def test_a_probability_outside_the_unit_interval_is_a_parse_error():
    client = _client(_reply({'type': 'noul', 'noul': 1.5}))

    outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert '1.5' in outcome.raw_content


@pytest.mark.parametrize(
    ('question', 'answer'),
    [
        pytest.param(_noul_question(), {'type': 'noul', 'noul': True}, id='noul-boolean'),
        pytest.param(_noul_question(), {'type': 'noul', 'noul': '0.9'}, id='noul-numeric-string'),
        pytest.param(
            ClassifyQuestion(kind='score', instructions='rate it', criteria=['bad', 'ok', 'good'], state='x'),
            {'type': 'score', 'score': True},
            id='score-boolean',
        ),
        pytest.param(
            ClassifyQuestion(kind='score', instructions='rate it', criteria=['bad', 'ok', 'good'], state='x'),
            {'type': 'score', 'score': '1.5'},
            id='score-numeric-string',
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_non_numeric_wire_value_is_a_parse_error(question: ClassifyQuestion, answer: dict[str, Any]):
    client = _client(_reply(answer))

    outcome = await _judge(client, question)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None


@pytest.mark.parametrize(
    'answer',
    [
        pytest.param({'type': 'choice', 'choice': 'A', 'confidence': '0.9'}, id='confidence-string'),
        pytest.param({'type': 'choice', 'choice': 'A', 'confidence': 1.1}, id='confidence-out-of-range'),
        pytest.param({'type': 'choice', 'choice': 'A', 'probabilities': {'A': '0.9'}}, id='probability-string'),
        pytest.param({'type': 'choice', 'choice': 'A', 'probabilities': {'A': -0.1}}, id='probability-out-of-range'),
    ],
)
@pytest.mark.asyncio
async def test_invalid_distribution_metadata_is_a_parse_error(answer: dict[str, Any]):
    question = ClassifyQuestion(
        kind='choice',
        instructions='pick one',
        criteria={'A': 'first', 'B': 'second'},
        state='x',
    )
    client = _client(_reply(answer))

    outcome = await _judge(client, question)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None


@pytest.mark.asyncio
async def test_a_score_off_the_level_scale_is_a_parse_error():
    question = ClassifyQuestion(
        kind='score',
        instructions='rate it',
        criteria=['useless', 'fine', 'excellent'],
        state={'answer': 'x'},
    )
    client = _client(_reply({'type': 'score', 'score': -1.0}))

    outcome = await _judge(client, question)

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert '-1' in outcome.raw_content


@pytest.mark.asyncio
async def test_a_reply_that_does_not_validate_is_a_parse_error_carrying_the_body(caplog: pytest.LogCaptureFixture):
    """The unreadable body reaches the outcome: '{}' would describe a call that never happened."""
    client = _client({'answers': 'not-a-mapping', 'model': 'jev-latest'})

    with caplog.at_level(logging.ERROR, logger='evaluatorq.common.judge'):
        outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is JudgeError.PARSE
    assert outcome.payload is None
    assert outcome.endpoint == 'classify'
    assert 'not-a-mapping' in outcome.raw_content
    assert 'not-a-mapping' in caplog.text


@pytest.mark.asyncio
async def test_a_reply_without_usage_stays_unpriced_and_unrecorded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Zeros on the span would read as a genuinely free call; unknown must stay unknown."""
    recorded: list[Any] = []
    # Patched on both modules: `record_llm_response` reads the name off `tracing`,
    # and the classify executor — which records usage itself, so that an unreadable
    # block records nothing rather than zeros — reads its own import.
    monkeypatch.setattr(tracing, 'record_token_usage', lambda span, **kw: recorded.append(kw))  # noqa: ARG005
    monkeypatch.setattr(llm_call, 'record_token_usage', lambda span, **kw: recorded.append(kw))  # noqa: ARG005
    client = _client({'answers': {'verdict': {'type': 'noul', 'noul': 0.9}}, 'model': 'jev-latest'})

    with caplog.at_level(logging.WARNING, logger='evaluatorq.common.llm_call'):
        outcome = await _judge(client, _noul_question())

    assert outcome.error_kind is None
    assert outcome.token_usage is not None
    assert outcome.token_usage.calls == 1
    assert outcome.token_usage.priced_calls == 0
    assert outcome.token_usage.total_cost is None
    assert recorded == []
    assert 'no readable usage block' in caplog.text
    assert JEV in caplog.text


@pytest.mark.parametrize(
    ('usage_block', 'case'),
    [
        pytest.param({}, 'empty', id='usage-empty'),
        pytest.param({'garbage': 1}, 'unreadable', id='usage-unreadable'),
    ],
)
@pytest.mark.asyncio
async def test_a_usage_block_that_does_not_parse_leaves_no_usage_attribute_on_the_span(
    usage_block: dict[str, Any],
    case: str,
    caplog: pytest.LogCaptureFixture,
):
    """Present-but-unreadable must read the same as absent: unpriced, not free."""
    span = MagicMock()
    client = _client({
        'answers': {'verdict': {'type': 'noul', 'noul': 0.9}},
        'model': 'jev-latest',
        'usage': usage_block,
    })

    with caplog.at_level(logging.WARNING, logger='evaluatorq.common.llm_call'):
        _payload, usage = await llm_call.execute_classify(
            client=client,
            model=JEV,
            question=_noul_question(),
            span=span,
            timeout_s=30.0,
            inject_trace_headers=False,
        )

    assert usage is not None, case
    assert usage.calls == 1
    assert usage.priced_calls == 0
    assert usage.total_cost is None
    assert 'no readable usage block' in caplog.text
    set_attrs = [call.args[0] for call in span.set_attribute.call_args_list]
    assert not [name for name in set_attrs if name.startswith('gen_ai.usage.')]
    # The rest of the response recording still happens — only usage is withheld.
    assert 'gen_ai.response.model' in set_attrs


@pytest.mark.asyncio
async def test_a_readable_usage_block_still_lands_on_the_span():
    """The withholding above is about unreadable usage, not about classify calls."""
    span = MagicMock()
    client = _client(_reply({'type': 'noul', 'noul': 0.9}, total_cost=0.004))

    _payload, usage = await llm_call.execute_classify(
        client=client,
        model=JEV,
        question=_noul_question(),
        span=span,
        timeout_s=30.0,
        inject_trace_headers=False,
    )

    assert usage is not None
    set_attrs = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert set_attrs['gen_ai.usage.input_tokens'] == 120
    assert set_attrs['gen_ai.usage.total_cost'] == 0.004


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
