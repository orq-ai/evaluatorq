"""Group D trajectory-tag behavior and dependency selection."""

from __future__ import annotations

from evaluatorq.signals import SignalsConfig, TagThresholds, compute_signals
from evaluatorq.signals.models import SignalResult
from evaluatorq.signals.registry import SIGNALS
from evaluatorq.signals.tags import Clause, TagRule, _tag_signal
from evaluatorq.signals.walk import SignalContext

from .conftest import agent, bare, call, failed, traj, user


def _thresholds(**updates: float) -> TagThresholds:
    bundled = TagThresholds.bundled()
    return bundled.model_copy(update={'thresholds': {**bundled.thresholds, **updates}})


def test_tool_churn_fires_with_metric_evidence_and_reason() -> None:
    trajectory = traj(
        [user(), agent(calls=[call('Bash', {'cmd': 'pwd'})])],
    )
    config = SignalsConfig(
        tag_thresholds=_thresholds(**{
            'tool_churn.tool_call_count': 0,
            'tool_churn.distinct_tool_arg_ratio': 1.1,
            'tool_churn.tool_retry_count': 100,
        })
    )

    report = compute_signals(trajectory, config, only=['tool_churn'])

    tag = report.results['tool_churn']
    assert tag.value is True
    assert tag.group == 'D'
    assert tag.reason is not None and 'tool_call_count > 0' in tag.reason
    assert tag.evidence
    assert tag.rule_version == config.version
    assert 'tool_call_count' in report.results
    assert 'distinct_tool_arg_ratio' in report.results


def test_tool_churn_does_not_fire_when_clauses_are_below_thresholds() -> None:
    trajectory = traj([user(), agent(calls=[call('Bash', {'cmd': 'pwd'})])])
    config = SignalsConfig(
        tag_thresholds=_thresholds(**{
            'tool_churn.tool_call_count': 100,
            'tool_churn.distinct_tool_arg_ratio': 0,
            'tool_churn.tool_retry_count': 100,
        })
    )

    tag = compute_signals(trajectory, config, only=['tool_churn']).results['tool_churn']

    assert tag.value is False
    assert tag.no_basis is None
    assert tag.reason is None


def test_tag_has_no_basis_when_all_alternatives_are_unavailable() -> None:
    trajectory = traj([agent('answer without a user prompt')])

    tag = compute_signals(trajectory, only=['long_autonomous_run']).results['long_autonomous_run']

    assert tag.value is None
    assert tag.no_basis is not None
    assert 'required metrics' in tag.no_basis


def test_false_tool_churn_gate_determines_false_with_unknown_retry_metric() -> None:
    trajectory = traj([
        agent(calls=[call('fetch', {'q': 'first'}, 'first')], results=[bare('result')]),
        agent(calls=[call('fetch', {'q': 'second'}, 'second')], results=[bare('result')]),
    ])
    config = SignalsConfig(tag_thresholds=_thresholds(**{
        'tool_churn.tool_call_count': 100,
        'tool_churn.distinct_tool_arg_ratio': 0,
    }))

    report = compute_signals(trajectory, config, only=['tool_churn'])
    tag = report.results['tool_churn']

    assert report.results['tool_retry_count'].no_basis is not None
    assert tag.value is False
    assert tag.no_basis is None
    assert any('tool_call_count' in pc.name and pc.met is False for pc in tag.preconditions)


def _synthetic_tag(rule: TagRule, values: dict[str, float | None]) -> SignalResult:
    results = {
        name: (
            SignalResult(name=name, group='B', value=value)
            if value is not None
            else SignalResult(name=name, group='B', no_basis='test metric unavailable')
        )
        for name, value in values.items()
    }
    config = SignalsConfig()
    context = SignalContext.build(traj([agent()]), config)
    context.results.update(results)
    return _tag_signal(rule)(context)


def test_tag_resolution_uses_upper_bound_for_unknown_alternatives() -> None:
    rule = TagRule(
        'synthetic_rule',
        'synthetic test rule',
        gate=(Clause('gate', '>', fixed=0),),
        any_of=(Clause('first', '>', fixed=0), Clause('second', '>', fixed=0)),
        need=2,
    )

    impossible = _synthetic_tag(rule, {'gate': None, 'first': 1, 'second': 0})
    still_possible = _synthetic_tag(rule, {'gate': None, 'first': 1, 'second': None})
    alternatives_met_gate_unknown = _synthetic_tag(rule, {'gate': None, 'first': 1, 'second': 2})

    assert impossible.value is False
    assert impossible.no_basis is None
    assert still_possible.value is None
    assert still_possible.no_basis is not None
    assert alternatives_met_gate_unknown.value is None
    assert alternatives_met_gate_unknown.no_basis is not None


def test_false_mandatory_gate_overrides_unknown_alternatives() -> None:
    rule = TagRule(
        'synthetic_rule',
        'synthetic test rule',
        gate=(Clause('gate', '>', fixed=0),),
        any_of=(Clause('first', '>', fixed=0), Clause('second', '>', fixed=0)),
    )

    tag = _synthetic_tag(rule, {'gate': 0, 'first': None, 'second': 0})

    assert tag.value is False
    assert tag.no_basis is None


def test_tag_percentile_override_uses_configured_cohort_table() -> None:
    trajectory = traj([user(), agent('working'), agent('finished')])
    config = SignalsConfig(tag_percentiles={'long_autonomous_run.max_autonomous_steps': 0})

    report = compute_signals(trajectory, config, only=['long_autonomous_run'])
    tag = report.results['long_autonomous_run']

    assert tag.value is True
    assert tag.rule_version == f'{config.version}'
    assert 'cohort p0' in (tag.reason or '')


def test_fixed_clause_uses_its_declared_threshold() -> None:
    trajectory = traj([
        agent(calls=[call('fetch', {}, f'failed-{index}')], results=[
            failed()
        ])
        for index in range(3)
    ])
    thresholds = _thresholds(**{'error_heavy.tool_error_count': 100})
    config = SignalsConfig(
        tag_thresholds=thresholds,
        tag_percentiles={'error_heavy.tool_error_count': 0},
    )

    tag = compute_signals(trajectory, config, only=['error_heavy']).results['error_heavy']

    assert tag.value is True
    assert 'tool_error_count > 2' in (tag.reason or '')


def test_only_tag_computes_each_dependency_once(monkeypatch) -> None:
    trajectory = traj([user(), agent(calls=[call('Bash', {'cmd': 'pwd'})])])
    original = SIGNALS['tool_call_count'][1]
    count = 0

    def counted(ctx):
        nonlocal count
        count += 1
        return original(ctx)

    patched = dict(SIGNALS)
    patched['tool_call_count'] = ('A', counted)
    monkeypatch.setattr('evaluatorq.signals.registry.SIGNALS', patched)

    report = compute_signals(trajectory, only=['tool_churn'])

    assert count == 1
    assert 'tool_call_count' in report.results
    assert 'tool_retry_count' in report.results
    assert list(report.results).index('tool_call_count') < list(report.results).index('tool_churn')
