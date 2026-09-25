"""Unit tests for `evaluatorq.insights.labeling` — the per-trace `/classify` label pass."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from loguru import logger

from evaluatorq.common.judge import (
    ClassifyAnswer,
    ClassifyOutcome,
    ClassifyQuestion,
    ClassifyRequest,
    ClassifyResponse,
    JudgeError,
)
from evaluatorq.insights.labeling import MATCH_KEY, label_traces
from evaluatorq.insights.models import LabelSpec
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
        compiled=INTENT_MATCH,
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
        compiled=INTENT_MATCH,
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
        compiled=INTENT_MATCH,
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
