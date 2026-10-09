"""Unit tests for `evaluatorq.insights.labeling` — the per-trace `/classify` label pass."""

from __future__ import annotations

import asyncio
import json
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

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
        assert MATCH_KEY in request.questions
        assert 'sentiment' in request.questions
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={
                    MATCH_KEY: ClassifyAnswer(type='choice', choice='billing', confidence=0.8),
                    'sentiment': ClassifyAnswer(
                        type='choice', choice='positive', confidence=0.9, probabilities={'positive': 0.9}
                    ),
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

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
        [make_trace('t1')],
        labels=[],
        compiled=(INTENT_MATCH, second),
        client=_client(),
        model='typesafe/jev-latest',
    )

    assert outcomes[0].matched is False
    assert outcomes[0].answers == {}


@pytest.mark.asyncio
async def test_missing_label_answer_fails_only_that_label(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
        await label_traces(
            [], labels=[], compiled=None, client=_client(), model='typesafe/jev-latest', parallelism=parallelism
        )


@pytest.mark.asyncio
async def test_unexpected_trace_failure_is_isolated_and_progress_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    original_messages = labeling_module.prompt_messages

    def sometimes_fail(trace):
        if trace.trace_id == 'bad':
            raise ValueError('malformed trace')
        return original_messages(trace)

    monkeypatch.setattr(labeling_module, 'prompt_messages', sometimes_fail)

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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

    async def fake_run_classify(
        *, client: Any, model: str, cfg: Any, request: ClassifyRequest, **_: Any
    ) -> ClassifyOutcome:
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
        [make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='typesafe/jev-latest', coding=True
    )

    asked = [set(request.questions) for request in calls]
    assert {'coding_agent'} in asked
    assert {'sentiment', 'task_type', 'outcome', 'verified', 'scope_creep', 'user_corrections'} in asked
    assert {'unfixed_error', 'risky_action'} in asked
    assert outcome.answers['coding_agent'].value is True
    assert outcome.answers['task_type'].value == 'bugfix'
    assert outcome.answers['risky_action'].value == 'none'
    assert outcome.error is None


_SECRET = 'sk-live-0123456789abcdef'


def _shell_trace(trace_id: str) -> TraceRecord:
    return make_trace(trace_id).model_copy(
        update={
            'messages': (
                {'role': 'user', 'content': 'Deploy it.'},
                {
                    'role': 'assistant',
                    'content': 'CONVERSATION_ONLY this belongs to the main conversation',
                    'tool_calls': [
                        {
                            'id': 'c1',
                            'type': 'function',
                            'function': {'name': 'Bash', 'arguments': '{"command": "make"}'},
                        }
                    ],
                },
                {'role': 'tool', 'tool_call_id': 'c1', 'content': f'Exit code 1\nAPI_KEY={_SECRET}\n2 failed'},
            )
        }
    )


def _tool_activity_states(calls: list[ClassifyRequest]) -> list[dict[str, Any]]:
    return [
        request.state for request in calls if 'unfixed_error' in request.questions and isinstance(request.state, dict)
    ]


@pytest.mark.asyncio
async def test_shell_output_reaches_the_tool_activity_question_only_after_scrubbing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluatorq.insights import labeling as labeling_module

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.9)}))
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))
    await label_traces(
        [_shell_trace('t')], labels=[], compiled=None, client=_client(), model='typesafe/jev-latest', coding=True
    )

    [state] = _tool_activity_states(calls)
    [activity] = [entry for entry in state['messages'] if entry['type'] == 'tool_call']
    [result] = activity['results']
    assert 'CONVERSATION_ONLY' not in json.dumps(state)
    assert activity['input'] == 'make'
    assert result['text'] == 'Exit code 1\nAPI_KEY=<API_KEY>\n2 failed'
    assert _SECRET not in json.dumps(state)
    assert 'nonzero_exit' in result.get('output_markers', [])

    gate = next(request for request in calls if 'coding_agent' in request.questions)
    gate_state = str(gate.state)
    assert 'Tool calls: 1' in gate_state
    assert 'make x1' in gate_state
    assert 'CONVERSATION_ONLY' not in gate_state
    assert 'Exit code 1' not in gate_state


@pytest.mark.asyncio
async def test_non_jev_request_reserves_serialized_questions_from_the_global_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluatorq.common.llm_call import classify_request_body
    from evaluatorq.common.model_input import serialized_chars

    model = 'openai/gpt-6-luna'
    trace = make_trace('non-jev-question-cap').model_copy(
        update={'messages': ({'role': 'user', 'content': 'REQUEST-START-' + 'x' * 10_000 + '-REQUEST-END'},)}
    )
    label = LabelSpec(
        name='cap_test',
        kind='choice',
        instructions='Classify this request. ' + 'q' * 500,
        criteria={'resolved': 'Resolved.', 'unresolved': 'Unresolved.'},
    )
    requests: list[ClassifyRequest] = []

    async def fake_run_classify(*, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        requests.append(request)
        return ClassifyOutcome(
            response=ClassifyResponse(
                answers={'cap_test': ClassifyAnswer(type='choice', choice='resolved')}
            )
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)
    [outcome] = await label_traces(
        [trace],
        labels=[label],
        compiled=None,
        client=_client(),
        model=model,
        trace_input_chars=1_500,
    )

    [request] = requests
    assert outcome.error is None
    assert isinstance(request.state, str)
    question_chars = serialized_chars(classify_request_body(model, '', request.questions)['questions'])
    assert len(request.state) + question_chars <= 1_500
    assert 'REQUEST-START-' in request.state
    assert 'REQUEST-END' in request.state


@pytest.mark.asyncio
async def test_questions_over_the_global_cap_fail_before_a_classifier_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = LabelSpec(name='too_large', kind='noul', instructions='q' * 1_000)
    requests: list[ClassifyRequest] = []

    async def fake_run_classify(*, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        requests.append(request)
        return ClassifyOutcome(response=ClassifyResponse(answers={}))

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)
    [outcome] = await label_traces(
        [make_trace('question-over-cap')],
        labels=[label],
        compiled=None,
        client=_client(),
        model='openai/gpt-6-luna',
        trace_input_chars=256,
    )

    assert not requests
    assert outcome.error is not None
    assert 'classifier questions leave no state budget' in outcome.error

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('trace_input_chars', 'question_size', 'call_count'),
    [(4_096, 0, 12), (500_000, 0, 250), (100_000, 55_000, 250)],
)
async def test_jev_tool_requests_keep_exact_edges_and_both_input_budgets(
    monkeypatch: pytest.MonkeyPatch, trace_input_chars: int, question_size: int, call_count: int
) -> None:
    from evaluatorq.common.model_input import (
        JEV_STATE_ALL_QUESTIONS_CHARS,
        JEV_STATE_QUESTION_CHARS,
        JEV_STATE_CHARS,
        classifier_question_wire_payloads,
        serialized_chars,
        serialized_question_chars,
    )
    from evaluatorq.insights.presets import UNFIXED_ERROR

    trace_messages: list[dict[str, Any]] = [{'role': 'user', 'content': 'Deploy this safely.'}]
    tool_calls = []
    tool_results = []
    commands: dict[int, str] = {}
    outputs: dict[int, str] = {}
    for index in range(call_count):
        call_id = f'c{index}'
        command = f'printf HEAD-{index} ' + 'i' * 5_000 + f' --tail-arg=TAIL-{index}'
        output = f'OUT-HEAD-{index}' + 'o' * 5_000 + f' OUT-TAIL-{index}\nScript failed'
        commands[index] = command
        outputs[index] = output
        tool_calls.append({
            'id': call_id,
            'type': 'function',
            'function': {'name': 'Bash', 'arguments': json.dumps({'command': command})},
        })
        tool_results.append({
            'role': 'tool',
            'tool_call_id': call_id,
            'status': 'completed',
            'exit_code': 1,
            'error_code': f'ERR_{index}',
            'content': output,
        })
    trace_messages.extend([{'role': 'assistant', 'tool_calls': tool_calls}, *tool_results])
    trace = make_trace('jev-tool-budget').model_copy(update={'messages': tuple(trace_messages)})
    question = UNFIXED_ERROR.model_copy(update={'instructions': 'Classify tool activity. ' + 'q' * question_size})
    calls: list[ClassifyRequest] = []
    monkeypatch.setattr('evaluatorq.insights.labeling.CODING_TOOL_LABELS', (question,))
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.9)}))
    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', _coding_fake(detect, calls))

    await label_traces(
        [trace],
        labels=[],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
        coding=True,
        coding_labels=[question],
        trace_input_chars=trace_input_chars,
    )

    requests = [request for request in calls if 'unfixed_error' in request.questions]
    assert len(requests) > 1
    states = []
    for request in requests:
        assert isinstance(request.state, dict)
        state_chars = serialized_chars(request.state)
        question_payloads = classifier_question_wire_payloads(request.questions)
        longest_question = max(serialized_chars(item) for item in question_payloads.values())
        question_chars = serialized_question_chars(request.questions)
        assert state_chars + question_chars <= trace_input_chars
        assert state_chars <= JEV_STATE_CHARS
        assert state_chars + longest_question <= JEV_STATE_QUESTION_CHARS
        assert state_chars + question_chars <= JEV_STATE_ALL_QUESTIONS_CHARS
        states.append(request.state)

    entries = [entry for state in states for entry in state['messages']]
    activity = [entry for entry in entries if entry['type'] == 'tool_call']
    assert len(activity) == call_count
    assert {entry['id'] for entry in activity} == {f'c{index}' for index in range(call_count)}
    for entry in activity:
        index = int(entry['id'][1:])
        command = commands[index]
        assert f'TAIL-{index}' in entry['input']
        assert f'[... {len(command) - 200} chars left out ...]' in entry['input']
        [result] = entry['results']
        assert result['error_code'] == f'ERR_{index}'
        assert result['status'] == 'completed'
        assert result['exit_code'] == '1'
        assert f'OUT-TAIL-{index}' in result['text']
        assert f'[... {len(outputs[index]) - 200} chars left out ...]' in result['text']
        assert 'nonzero_exit' in result.get('output_markers', [])
        assert 'i' * 400 not in entry['input']
        assert 'o' * 400 not in result['text']


def test_jev_activity_pairs_repeated_ids_and_preserves_orphan_results() -> None:
    from evaluatorq.insights.transcript import tool_activity_chunks

    messages = (
        {'role': 'user', 'content': 'Check these command results.'},
        {'role': 'tool', 'tool_call_id': 'missing', 'content': 'orphan result'},
        {
            'role': 'assistant',
            'tool_calls': [
                {'id': 'duplicate', 'function': {'name': 'Bash', 'arguments': '{"command":"first"}'}},
                {'id': 'duplicate', 'function': {'name': 'Bash', 'arguments': '{"command":"second"}'}},
                {'id': 'parallel', 'function': {'name': 'Bash', 'arguments': '{"command":"parallel"}'}},
            ],
        },
        {'role': 'tool', 'tool_call_id': 'duplicate', 'content': 'first result'},
        {'role': 'tool', 'tool_call_id': 'duplicate', 'content': 'second result'},
        {'role': 'tool', 'tool_call_id': 'parallel', 'content': 'parallel result'},
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'duplicate', 'function': {'name': 'Bash', 'arguments': '{"command":"third"}'}}],
        },
        {'role': 'tool', 'tool_call_id': 'duplicate', 'content': 'third result'},
    )
    trace = make_trace('jev-duplicate-tools').model_copy(update={'messages': messages})

    [state] = tool_activity_chunks(trace, model='typesafe/jev-latest', trace_input_chars=10_000)

    assert isinstance(state, dict)
    entries = state['messages']
    activity = [entry for entry in entries if entry['type'] == 'tool_call']
    assert [entry['input'] for entry in activity] == ['first', 'second', 'parallel', 'third']
    assert activity[0]['results'][0]['text'] == 'first result'
    assert activity[1]['results'][0]['text'] == 'second result'
    assert activity[2]['results'][0]['text'] == 'parallel result'
    assert activity[3]['results'][0]['text'] == 'third result'
    assert any(entry['type'] == 'orphan_result' and entry['text'] == 'orphan result' for entry in entries)

def test_jev_activity_keeps_orphan_results_without_assistant_calls() -> None:
    from evaluatorq.insights.transcript import tool_activity_chunks

    trace = make_trace('orphan-only-tools').model_copy(
        update={'messages': ({'role': 'tool', 'tool_call_id': 'missing', 'content': 'orphan result'},)}
    )

    [state] = tool_activity_chunks(trace, model='typesafe/jev-latest', trace_input_chars=10_000)

    assert isinstance(state, dict)
    [entry] = state['messages']
    assert entry['type'] == 'orphan_result'
    assert entry['text'] == 'orphan result'

def test_jev_activity_uses_no_calls_sentinel_only_for_empty_activity() -> None:
    from evaluatorq.insights.transcript import tool_activity_chunks

    [chunk] = tool_activity_chunks(
        make_trace('no-tool-activity'), model='typesafe/jev-latest', trace_input_chars=10_000
    )

    assert chunk == 'No tool calls.'


@pytest.mark.asyncio
async def test_selected_coding_labels_keep_gate_and_only_ask_selected_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module
    from evaluatorq.insights.presets import TASK_TYPE, UNFIXED_ERROR

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.9)}))
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))

    [outcome] = await label_traces(
        [make_trace('t')],
        labels=[],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
        coding=True,
        coding_labels=[TASK_TYPE, UNFIXED_ERROR],
    )

    assert [set(request.questions) for request in calls] == [{'coding_agent'}, {'task_type'}, {'unfixed_error'}]
    assert set(outcome.answers) == {'coding_agent', 'task_type', 'unfixed_error'}


@pytest.mark.asyncio
async def test_conversation_only_coding_subset_skips_empty_tool_classify(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module
    from evaluatorq.insights.presets import TASK_TYPE

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.9)}))
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))

    [outcome] = await label_traces(
        [make_trace('t')],
        labels=[],
        compiled=None,
        client=_client(),
        model='typesafe/jev-latest',
        coding=True,
        coding_labels=[TASK_TYPE],
    )

    assert [set(request.questions) for request in calls] == [{'coding_agent'}, {'task_type'}]
    assert set(outcome.answers) == {'coding_agent', 'task_type'}


@pytest.mark.asyncio
async def test_non_coding_agent_is_not_asked_coding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module

    calls: list[ClassifyRequest] = []
    detect = ClassifyOutcome(response=ClassifyResponse(answers={'coding_agent': ClassifyAnswer(type='noul', noul=0.1)}))
    monkeypatch.setattr(labeling_module, 'run_classify', _coding_fake(detect, calls))

    [outcome] = await label_traces(
        [make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='typesafe/jev-latest', coding=True
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
            [make_trace('t')],
            labels=[SENTIMENT],
            compiled=None,
            client=_client(),
            model='typesafe/jev-latest',
            coding=True,
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

    await label_traces(
        [make_trace('t')], labels=[SENTIMENT], compiled=None, client=_client(), model='typesafe/jev-latest'
    )

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

    assert (
        _merge_chunks('risky_action', [_choice('none'), _choice('deleted', 0.6), _choice('none', 0.9)]).value
        == 'deleted'
    )
    assert _merge_chunks('risky_action', [_choice('none', 0.6), _choice('none', 0.9)]).confidence == 0.9
    # A failed chunk may have held the finding, so with no finding elsewhere the label fails.
    assert _merge_chunks('risky_action', [_choice('none'), _choice(None, error='boom')]).error == 'boom'
    assert _merge_chunks('risky_action', [_choice('published'), _choice(None, error='boom')]).value == 'published'


def test_merge_chunks_last_chunk_decides_unfixed_error() -> None:
    from evaluatorq.insights.labeling import _merge_chunks

    assert _merge_chunks('unfixed_error', [_noul(0.9), _noul(0.1)]).value is False
    assert _merge_chunks('unfixed_error', [_noul(0.1), _noul(None, 'boom')]).error == 'boom'


@pytest.mark.asyncio
@pytest.mark.parametrize(('model', 'structured'), [('typesafe/jev-latest', True), ('openai/gpt-6-luna', False)])
async def test_classifier_uses_jev_state_only_for_jev_and_full_transcript_otherwise(
    monkeypatch: pytest.MonkeyPatch, model: str, structured: bool
) -> None:
    messages = (
        {'role': 'user', 'content': 'REQUEST-START please help.'},
        {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': 'lookup-1',
                    'type': 'function',
                    'function': {'name': 'search', 'arguments': '{"query":"middle-input"}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'lookup-1', 'content': 'MIDDLE-RESULT found one record.'},
        {'role': 'assistant', 'content': 'FINAL-ANSWER here is the result.'},
    )
    trace = make_trace('bounded').model_copy(update={'messages': messages})
    requests: list[ClassifyRequest] = []

    async def fake_run_classify(*, request: ClassifyRequest, **_: Any) -> ClassifyOutcome:
        requests.append(request)
        return ClassifyOutcome(
            response=ClassifyResponse(answers={'sentiment': ClassifyAnswer(type='choice', choice='positive')})
        )

    monkeypatch.setattr('evaluatorq.insights.labeling.run_classify', fake_run_classify)

    await label_traces(
        [trace],
        labels=[SENTIMENT],
        compiled=None,
        client=_client(),
        model=model,
        trace_input_chars=100_000,
    )

    state = requests[0].state
    if structured:
        assert isinstance(state, dict)
        assert state['messages'][0]['text'] == 'REQUEST-START please help.'
        assert any(entry['type'] == 'tool_call' for entry in state['messages'])
        assert any(
            entry['type'] == 'message' and entry['text'] == 'FINAL-ANSWER here is the result.'
            for entry in state['messages']
        )
    else:
        assert isinstance(state, str)
        for phrase in ('REQUEST-START', 'middle-input', 'MIDDLE-RESULT', 'FINAL-ANSWER'):
            assert phrase in state
