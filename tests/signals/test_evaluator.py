"""Evaluator adapters for deterministic trajectory signals."""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluatorq.contracts import Message
from evaluatorq.formats import AtifTrajectory, ChatConversation, OtelTrace, ResponsesConversation
from evaluatorq.signals import signal_evaluator, signal_evaluators, to_trajectory
from evaluatorq.signals.evaluator import compute_signals as evaluator_compute
from evaluatorq.signals.models import Evidence, SignalReport, SignalResult
from evaluatorq.types import EvaluationResult, Evaluator, ScorerParameter

from .conftest import agent, call, failed, traj, user


async def _score(evaluator: Evaluator, output: Any) -> EvaluationResult:
    scorer = evaluator['scorer']
    value = await scorer(cast(ScorerParameter, cast(object, {'output': output})))
    return EvaluationResult.model_validate(value)


def test_to_trajectory_accepts_supported_inputs() -> None:
    trajectory = traj([user(), agent()])
    messages = [Message(role='user', content='hi'), Message(role='assistant', content='hello')]
    responses = ResponsesConversation(items=[{'role': 'user', 'content': 'hi'}])
    otel = OtelTrace.from_orq([
        {
            'span_id': 's1',
            'name': 'chat',
            'attributes': {
                'gen_ai.operation.name': 'chat',
                'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}],
                'gen_ai.output.messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hello'}]}],
            },
        }
    ])

    assert to_trajectory(trajectory) is trajectory
    assert to_trajectory(ChatConversation(messages=messages)) is not None
    assert to_trajectory(responses) is not None
    assert to_trajectory(otel) is not None
    assert to_trajectory([message.model_dump(exclude_none=True) for message in messages]) is not None
    assert to_trajectory({'messages': messages}) is not None


def test_to_trajectory_returns_none_for_unsupported_or_malformed_output() -> None:
    assert to_trajectory('not a trajectory') is None
    assert to_trajectory({'messages': 'not a message list'}) is None
    assert to_trajectory([{'invalid': 'message'}]) is None


@pytest.mark.asyncio
async def test_unconvertible_output_is_an_inconclusive_evaluation(caplog: pytest.LogCaptureFixture) -> None:
    result = await _score(signal_evaluator('tool_error_count'), {'unexpected': True})

    assert result.value is None
    assert result.pass_ is None
    assert 'dict' in (result.explanation or '')
    assert 'unsupported output of type dict' in caplog.text


@pytest.mark.asyncio
async def test_no_basis_is_an_inconclusive_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_basis(*args: Any, **kwargs: Any) -> SignalReport:
        return SignalReport(
            trajectory_id='t',
            config_version='v',
            results={'tool_error_count': SignalResult(name='tool_error_count', group='B', no_basis='no calls')},
        )

    monkeypatch.setattr('evaluatorq.signals.evaluator.compute_signals', no_basis)
    result = await _score(signal_evaluator('tool_error_count'), traj([user()]))

    assert result.value is None
    assert result.pass_ is None
    assert result.explanation == 'No basis for tool_error_count: no calls'


@pytest.mark.asyncio
async def test_fired_tag_fails_and_explanation_limits_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = SignalResult(
        name='error_heavy',
        group='D',
        value=True,
        reason='too many errors',
        evidence=[Evidence(step_id=index, call_id=f'call-{index}') for index in range(1, 12)],
    )
    monkeypatch.setattr(
        'evaluatorq.signals.evaluator.compute_signals',
        lambda *args, **kwargs: SignalReport(trajectory_id='t', config_version='v', results={'error_heavy': signal}),
    )

    result = await _score(signal_evaluator('error_heavy'), traj([user()]))

    assert result.value is True
    assert result.pass_ is False
    assert 'too many errors' in (result.explanation or '')
    assert 'step 1 / call call-1' in (result.explanation or '')
    assert '(+1 more)' in (result.explanation or '')
    assert 'call-11' not in (result.explanation or '')


@pytest.mark.asyncio
async def test_unfired_tag_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = SignalResult(name='error_heavy', group='D', value=False)
    monkeypatch.setattr(
        'evaluatorq.signals.evaluator.compute_signals',
        lambda *args, **kwargs: SignalReport(trajectory_id='t', config_version='v', results={'error_heavy': signal}),
    )

    result = await _score(signal_evaluator('error_heavy'), traj([user()]))

    assert result.value is False
    assert result.pass_ is True


@pytest.mark.asyncio
async def test_count_threshold_uses_at_or_below_as_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = SignalResult(name='tool_error_count', group='B', value=3)
    monkeypatch.setattr(
        'evaluatorq.signals.evaluator.compute_signals',
        lambda *args, **kwargs: SignalReport(
            trajectory_id='t', config_version='v', results={'tool_error_count': signal}
        ),
    )

    evaluator = signal_evaluator('tool_error_count', threshold=3)
    result = await _score(evaluator, traj([user()]))

    assert result.value == 3
    assert result.pass_ is True
    assert (await _score(signal_evaluator('tool_error_count'), traj([user()]))).pass_ is None


@pytest.mark.asyncio
async def test_non_numeric_count_threshold_is_inconclusive_and_keeps_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = SignalResult(
        name='llm_call_count',
        group='A',
        value={'bash': 2},
        evidence=[Evidence(step_id=4, call_id='call-4')],
    )
    monkeypatch.setattr(
        'evaluatorq.signals.evaluator.compute_signals',
        lambda *args, **kwargs: SignalReport(trajectory_id='t', config_version='v', results={'llm_call_count': signal}),
    )

    result = await _score(
        signal_evaluator('llm_call_count', threshold=3),
        traj([user()]),
    )

    assert result.value == {'bash': 2}
    assert result.pass_ is None
    assert 'non-numeric value' in (result.explanation or '')
    assert 'step 4 / call call-4' in (result.explanation or '')


def test_signal_evaluators_select_tags_groups_all_and_names() -> None:
    tags = signal_evaluators()
    explicit_tags = signal_evaluators('tags')
    assert tags
    assert [item['name'] for item in explicit_tags] == [item['name'] for item in tags]
    assert all(
        name['name']
        in {
            'long_autonomous_run',
            'delegation_heavy',
            'error_heavy',
            'tool_churn',
            'tool_loop',
            'stalled',
            'output_heavy',
            'inefficient_execution',
        }
        for name in tags
    )
    assert len(signal_evaluators('all')) > len(tags)
    assert {item['name'] for item in signal_evaluators('D')} == {item['name'] for item in tags}
    assert len(signal_evaluators('A')) > 0
    assert [item['name'] for item in signal_evaluators(['error_heavy', 'tool_error_count'])] == [
        'error_heavy',
        'tool_error_count',
    ]
    with pytest.raises(ValueError, match='Unknown signal'):
        signal_evaluators(['not_a_signal'])


@pytest.mark.asyncio
async def test_tag_evaluator_requests_only_tag_and_computes_its_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluatorq.signals import registry
    from evaluatorq.signals.tags import TAG_DEPENDENCIES

    calls: list[list[str]] = []

    def recording_compute(trajectory: AtifTrajectory, *, config: Any, only: list[str]) -> SignalReport:
        calls.append(only)
        return evaluator_compute(trajectory, config=config, only=only)

    monkeypatch.setattr('evaluatorq.signals.evaluator.compute_signals', recording_compute)
    trajectory = traj([user(), agent(calls=[call('Bash')], results=[failed()]), agent()])
    await _score(signal_evaluator('error_heavy'), trajectory)

    assert calls == [['error_heavy']]
    expected = {'error_heavy', *TAG_DEPENDENCIES['error_heavy']}
    assert set(registry.compute_signals(trajectory, only=['error_heavy']).results) == expected
