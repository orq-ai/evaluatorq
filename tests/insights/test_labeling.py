"""Unit tests for `evaluatorq.insights.labeling` — the per-trace `/classify` label pass."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from loguru import logger
from openai import APIStatusError

from evaluatorq.common.judge import (
    ClassifyAnswer,
    ClassifyOutcome,
    ClassifyQuestion,
    ClassifyRequest,
    ClassifyResponse,
    JudgeError,
)
from evaluatorq.contracts import LLMCallConfig, Usage
from evaluatorq.insights.labeling import MATCH_KEY, label_traces
from evaluatorq.insights.models import LabelAnswer, LabelSpec
from evaluatorq.insights.usage import UsageLedger
from evaluatorq.trace_finder.models import CompiledQuery, TraceRecord, ValueSelection


def make_trace(trace_id: str) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=f'span-{trace_id}',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': 'hello'},),
        project='default',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
    )


SENTIMENT = LabelSpec(
    name='sentiment',
    kind='choice',
    instructions='Classify sentiment.',
    criteria={'positive': 'happy', 'negative': 'unhappy'},
)

MADE_ERRORS = LabelSpec(
    name='made_errors',
    kind='noul',
    instructions='Decide whether the assistant made an error.',
    criteria={'true': 'the assistant made an error', 'false': 'no assistant errors'},
)

INTENT_MATCH = CompiledQuery(
    task=ClassifyQuestion(
        kind='choice',
        instructions='Classify the support need.',
        criteria={'billing': 'Billing help.', 'technical': 'Technical help.'},
        state={},
    ),
    selection=ValueSelection(kind='values', values=('billing',)),
)


def _client() -> Any:
    return object()


def _api_status_error(status: int) -> APIStatusError:
    request = httpx.Request('POST', 'https://example.com')
    response = httpx.Response(status, request=request)
    return APIStatusError(message=f'HTTP {status}', response=response, body=None)


@pytest.mark.asyncio
async def test_label_retries_http_520_but_records_each_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def no_wait(*_: Any) -> None:
        return None

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ClassifyOutcome(
                error_kind=JudgeError.API_STATUS,
                error_exc=_api_status_error(520),
                token_usage=Usage(input_tokens=3, calls=1),
            )
        return ClassifyOutcome(
            response=ClassifyResponse(answers={'sentiment': ClassifyAnswer(type='choice', choice='positive')}),
            token_usage=Usage(input_tokens=5, calls=1),
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)
    monkeypatch.setattr('evaluatorq.common.retry.asyncio.sleep', no_wait)
    ledger = UsageLedger()
    results = await label_traces(
        [make_trace('520')],
        labels=[SENTIMENT],
        compiled=None,
        client=_client(),
        model='m',
        cfg=LLMCallConfig(model='m', retry_count=1),
        usage=ledger,
    )

    recorded = ledger.totals()['label']
    assert calls == 2
    assert results[0].error is None
    assert recorded is not None
    assert recorded.calls == 2
    assert recorded.input_tokens == 8


@pytest.mark.asyncio
async def test_label_does_not_retry_http_400(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        nonlocal calls
        calls += 1
        return ClassifyOutcome(
            error_kind=JudgeError.API_STATUS,
            error_exc=_api_status_error(400),
            token_usage=Usage(input_tokens=3, calls=1),
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)
    ledger = UsageLedger()
    results = await label_traces(
        [make_trace('400')],
        labels=[SENTIMENT],
        compiled=None,
        client=_client(),
        model='m',
        cfg=LLMCallConfig(model='m', retry_count=2),
        usage=ledger,
    )

    recorded = ledger.totals()['label']
    assert calls == 1
    assert results[0].error is not None
    assert recorded is not None
    assert recorded.calls == 1
    assert recorded.input_tokens == 3


@pytest.mark.asyncio
async def test_all_labels_answered_and_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        assert MATCH_KEY in request.questions
        assert 'sentiment' in request.questions
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={
                    MATCH_KEY: ClassifyAnswer(type='choice', choice='billing', confidence=0.8),
                    'sentiment': ClassifyAnswer(type='choice', choice='positive', confidence=0.9, probabilities={'positive': 0.9}),
                }
            )
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    trace = make_trace('t1')
    outcomes = await label_traces(
        [trace],
        labels=[SENTIMENT],
        compiled=(INTENT_MATCH,),
        client=_client(),
        model='typesafe/jev-latest',
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.error is None
    assert outcome.matched is True
    assert outcome.answers['sentiment'].value == 'positive'
    assert outcome.answers['sentiment'].confidence == 0.9
    assert outcome.answers['sentiment'].probabilities == {'positive': 0.9}
    assert outcome.answers['sentiment'].error is None


@pytest.mark.asyncio
async def test_multiple_population_dimensions_must_all_match(monkeypatch: pytest.MonkeyPatch) -> None:
    second = INTENT_MATCH.model_copy(update={'name': 'customer sentiment'})

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        assert set(request.questions) == {'__match_0__', '__match_1__'}
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={
                    '__match_0__': ClassifyAnswer(type='choice', choice='billing'),
                    '__match_1__': ClassifyAnswer(type='choice', choice='technical'),
                }
            )
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)
    outcomes = await label_traces(
        [make_trace('t1')], labels=[], compiled=(INTENT_MATCH, second),
        client=_client(), model='typesafe/jev-latest',
    )

    assert outcomes[0].matched is False
    assert outcomes[0].answers == {}


@pytest.mark.asyncio
async def test_missing_label_answer_fails_only_that_label(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={
                    MATCH_KEY: ClassifyAnswer(type='choice', choice='billing'),
                    # 'sentiment' deliberately missing from the reply.
                }
            )
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    trace = make_trace('t1')
    outcomes = await label_traces(
        [trace],
        labels=[SENTIMENT],
        compiled=(INTENT_MATCH,),
        client=_client(),
        model='typesafe/jev-latest',
    )

    outcome = outcomes[0]
    assert outcome.error is None
    assert outcome.matched is True
    assert outcome.answers['sentiment'].error is not None
    assert outcome.answers['sentiment'].value is None


@pytest.mark.asyncio
async def test_outcome_failure_fails_every_answer_and_clears_match(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(error_kind=JudgeError.API_STATUS, error_message='500 from router')

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    trace = make_trace('t1')
    outcomes = await label_traces(
        [trace],
        labels=[SENTIMENT],
        compiled=(INTENT_MATCH,),
        client=_client(),
        model='typesafe/jev-latest',
    )

    outcome = outcomes[0]
    assert outcome.error is not None
    assert outcome.matched is None
    assert outcome.answers['sentiment'].error is not None
    assert outcome.answers['sentiment'].value is None


@pytest.mark.asyncio
async def test_no_labels_and_no_compiled_query_skips_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        nonlocal calls
        calls += 1
        raise AssertionError('run_classify must not be called')

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    trace = make_trace('t1')
    outcomes = await label_traces(
        [trace],
        labels=[],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
    )

    assert calls == 0
    outcome = outcomes[0]
    assert outcome.answers == {}
    assert outcome.matched is None
    assert outcome.error is None


@pytest.mark.asyncio
async def test_concurrency_never_exceeds_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return ClassifyOutcome(
            response=ClassifyResponse(answers={'sentiment': ClassifyAnswer(type='choice', choice='positive')})
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    traces = [make_trace(f't{i}') for i in range(20)]
    outcomes = await label_traces(
        traces,
        labels=[SENTIMENT],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
        parallelism=3,
    )

    assert len(outcomes) == 20
    assert max_in_flight <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize('parallelism', [0, -1])
async def test_parallelism_must_be_positive(parallelism: int) -> None:
    with pytest.raises(ValueError, match='parallelism must be greater than zero'):
        await label_traces([], labels=[], compiled=None, client=_client(), model='typesafe/jev-latest', parallelism=parallelism)


@pytest.mark.asyncio
async def test_unexpected_trace_failure_is_isolated_and_progress_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    original_view = labeling_module.conversation_view

    def sometimes_fail(trace):
        if trace.trace_id == 'bad':
            raise ValueError('malformed trace')
        return original_view(trace)

    monkeypatch.setattr(labeling_module, "conversation_view", sometimes_fail)

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(
            response=ClassifyResponse(answers={'sentiment': ClassifyAnswer(type='choice', choice='positive')})
        )

    monkeypatch.setattr(labeling_module, 'run_classify', fake_run_classify)
    progress: list[tuple[int, int]] = []
    outcomes = await label_traces(
        [make_trace('bad'), make_trace('good')],
        labels=[SENTIMENT],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
        on_progress=lambda completed, total: progress.append((completed, total)),
    )

    assert outcomes[0].error == 'malformed trace'
    assert outcomes[0].answers['sentiment'].error == 'malformed trace'
    assert outcomes[1].error is None
    assert outcomes[1].answers['sentiment'].value == 'positive'
    assert len(progress) == 2


@pytest.mark.asyncio
async def test_noul_label_keeps_bool_value_and_raw_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={'made_errors': ClassifyAnswer(type='noul', noul=0.82, confidence=0.6)},
            )
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    trace = make_trace('t1')
    outcomes = await label_traces(
        [trace],
        labels=[MADE_ERRORS],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
    )

    answer = outcomes[0].answers['made_errors']
    assert answer.error is None
    assert answer.value is True
    assert answer.confidence == 0.6
    assert answer.probabilities == {'true': 0.82, 'false': pytest.approx(0.18)}


@pytest.mark.asyncio
async def test_unreadable_answer_logs_a_warning_naming_trace_and_label(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        return ClassifyOutcome(
            response=ClassifyResponse(
                # 'choice' answer for a 'noul' question: unreadable, not missing.
                answers={'made_errors': ClassifyAnswer(type='choice', choice='true')},
            )
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    trace = make_trace('t1')
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        outcomes = await label_traces(
            [trace],
            labels=[MADE_ERRORS],
            compiled=None,
            client=_client(),
            model='typesafe/jev-latest',
        )
    finally:
        logger.remove(sink_id)

    answer = outcomes[0].answers['made_errors']
    assert answer.error is not None
    assert answer.value is None
    assert any('t1' in message and 'made_errors' in message and 'unreadable' in message for message in messages)


def _coding_fake(detect: ClassifyOutcome, calls: list[ClassifyRequest]) -> Any:
    """A `/classify` fake answering the coding check with `detect` and every other question with a fixed answer."""

    async def fake_run_classify(*, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        calls.append(request)
        if 'coding_agent' in request.questions:
            return detect
        answers: dict[str, ClassifyAnswer] = {}
        for name, question in request.questions.items():
            if question.kind == 'noul':
                answers[name] = ClassifyAnswer(type='noul', noul=0.8)
            elif question.kind == 'score':
                answers[name] = ClassifyAnswer(type='score', score=1.0)
            else:
                assert isinstance(question.criteria, dict)
                answers[name] = ClassifyAnswer(type='choice', choice=next(iter(question.criteria)))
        return ClassifyOutcome(response=ClassifyResponse(answers=answers))

    return fake_run_classify


@pytest.mark.asyncio
async def test_coding_agent_gets_coding_labels_in_conversation_and_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.9)}))
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))

    [outcome] = await label_traces(
        [make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='jev', coding=True
    )

    asked = [set(request.questions) for request in calls]
    assert {'coding_agent'} in asked
    assert {'sentiment', 'task_type', 'outcome', 'verified', 'scope_creep', 'user_corrections'} in asked
    assert {'unfixed_error', 'risky_action'} in asked
    assert outcome.answers['coding_agent'].value is True
    assert outcome.answers['task_type'].value == 'bugfix'
    assert outcome.answers['risky_action'].value == 'none'
    assert outcome.error is None


@pytest.mark.asyncio
async def test_non_coding_agent_is_not_asked_coding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.1)}))
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))

    [outcome] = await label_traces(
        [make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='jev', coding=True
    )

    assert [set(request.questions) for request in calls] == [{'coding_agent'}, {'sentiment'}]
    assert outcome.answers['coding_agent'].value is False
    assert 'task_type' not in outcome.answers


@pytest.mark.asyncio
async def test_failed_coding_check_skips_coding_labels_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(error_kind=JudgeError.PARSE, error_message='bad reply')
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING')
    try:
        [outcome] = await label_traces(
            [make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='jev', coding=True
        )
    finally:
        logger.remove(sink)

    assert outcome.answers['coding_agent'].error == 'bad reply'
    assert 'task_type' not in outcome.answers
    assert outcome.answers['sentiment'].value == 'positive'
    assert any('coding labels were not asked' in message for message in messages)


@pytest.mark.asyncio
async def test_coding_off_makes_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    calls: list[ClassifyRequest] = []
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(ClassifyOutcome(), calls))

    await label_traces([make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='jev')

    assert [set(request.questions) for request in calls] == [{'sentiment'}]


def _noul(true: float | None, error: str | None = None) -> LabelAnswer:
    if error is not None:
        return LabelAnswer(value=None, confidence=None, probabilities=None, error=error)
    assert true is not None
    return LabelAnswer(value=true >= 0.5, confidence=None, probabilities={'true': true, 'false': 1 - true}, error=None)


def _choice(value: str | None, confidence: float = 0.8, error: str | None = None) -> LabelAnswer:
    return LabelAnswer(value=value, confidence=None if error else confidence, probabilities=None, error=error)


def test_merge_chunks_any_finding_wins_for_risky_action() -> None:
    from evaluatorq.insights.labeling import _merge_chunks

    assert _merge_chunks('risky_action', [_choice('none'), _choice('deleted', 0.6), _choice('none', 0.9)]).value == 'deleted'
    assert _merge_chunks('risky_action', [_choice('none', 0.6), _choice('none', 0.9)]).confidence == 0.9
    # A failed chunk may have held the finding, so with no finding elsewhere the label fails.
    assert _merge_chunks('risky_action', [_choice('none'), _choice(None, error='boom')]).error == 'boom'
    assert _merge_chunks('risky_action', [_choice('published'), _choice(None, error='boom')]).value == 'published'


def test_merge_chunks_last_chunk_decides_unfixed_error() -> None:
    from evaluatorq.insights.labeling import _merge_chunks

    assert _merge_chunks('unfixed_error', [_noul(0.9), _noul(0.1)]).value is False
    assert _merge_chunks('unfixed_error', [_noul(0.1), _noul(None, 'boom')]).error == 'boom'
