"""Registry: failure isolation, `only` handling, and the report's no-basis rule."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from pydantic import ValidationError

from evaluatorq.signals import registry
from evaluatorq.signals.models import Evidence, Precondition, SignalReport, SignalResult, result
from evaluatorq.signals.registry import compute_signals

from .conftest import agent, call, context, ok, spawned, traj, user


def _fine(ctx):
    return result('fine', 'A', 3)


def _boom(ctx):
    msg = 'bad rule'
    raise RuntimeError(msg)


def _table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, 'SIGNALS', {'fine': ('A', _fine), 'boom': ('B', _boom)})


def test_the_registry_contains_group_a_and_reports_the_config_version() -> None:
    report = compute_signals(traj([user(), agent()], trajectory_id='t1'))
    assert 'max_depth' in report.results
    assert report.results['max_depth'].value == 0
    assert report.trajectory_id == 't1'
    assert report.config_version == 'v3-local-cc-2026-09-29'


def test_tool_activity_dependencies_are_registered_and_tags_inherit_them() -> None:
    assert registry.TOOL_ACTIVITY_DEPENDENT_SIGNALS <= set(registry.SIGNALS)
    assert registry._TAG_TOOL_DEPENDENCIES == {
        'long_autonomous_run', 'delegation_heavy', 'error_heavy', 'tool_churn', 'tool_loop', 'stalled',
        'output_heavy', 'inefficient_execution',
    }
    assert registry._TAG_TOOL_DEPENDENCIES <= set(registry.SIGNALS)


def test_every_signal_that_reads_calls_requires_complete_source_coverage() -> None:
    call_readers = set()
    for name, (_, signal) in registry.SIGNALS.items():
        tree = ast.parse(textwrap.dedent(inspect.getsource(signal)))
        reads_calls = any(
            isinstance(node, ast.Attribute)
            and node.attr == 'calls'
            and isinstance(node.value, ast.Name)
            and node.value.id == 'ctx'
            for node in ast.walk(tree)
        )
        if reads_calls:
            call_readers.add(name)

    assert call_readers <= registry.TOOL_ACTIVITY_DEPENDENT_SIGNALS


def test_subagent_invocation_count_has_no_basis_when_source_tool_activity_is_unmapped() -> None:
    child = traj([user('subtask'), agent('child answer')], trajectory_id='child')
    root = traj(
        [agent(
            calls=[call('delegate', call_id='delegate-1')],
            results=[spawned('child', call_id='delegate-1')],
            extra={'evaluatorq.responses_output_items': [{'type': 'mcp_call', 'name': 'delegate'}]},
        )],
        subagents=[child],
    )

    report = compute_signals(root, only=['subagent_invocation_count'])

    assert report.results['subagent_invocation_count'].value is None
    assert 'unrepresented: delegate @step 1' in (report.results['subagent_invocation_count'].no_basis or '')


def test_a_raising_signal_becomes_no_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    _table(monkeypatch)
    report = compute_signals(traj([user()]))
    assert report.results['fine'].value == 3
    assert report.results['boom'].no_basis == 'signal raised RuntimeError: bad rule'
    assert report.results['boom'].group == 'B'
    assert report.values() == {'fine': 3}


def test_only_selects_signals_and_unknown_names_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _table(monkeypatch)
    assert list(compute_signals(traj([user()]), only=['boom']).results) == ['boom']
    with pytest.raises(ValueError, match='nope'):
        compute_signals(traj([user()]), only=['fine', 'nope'])


def test_merge_refuses_a_duplicate_name() -> None:
    with pytest.raises(ValueError, match='twice'):
        registry._merge({'a': ('A', _fine)}, {'a': ('B', _fine)})


def test_no_basis_with_a_value_is_invalid() -> None:
    with pytest.raises(ValidationError):
        SignalResult(name='x', group='A', value=1, no_basis='why')


def test_result_turns_a_failed_required_precondition_into_no_basis() -> None:
    pre = [
        Precondition(name='a', met=False, detail='none'),
        Precondition(name='b', met=False, detail='x', required=False),
    ]
    built = result('x', 'A', 5, preconditions=pre)
    assert built.value is None
    assert built.no_basis == 'a: none'
    assert built.preconditions == pre
    assert result('x', 'A', 5, preconditions=pre[1:]).value == 5
    assert result('x', 'A', 5, preconditions=[Precondition(name='p', met='partial')]).value == 5


def test_result_sorts_evidence_by_path_then_step() -> None:
    ev = [Evidence(step_id=3), Evidence(step_id=1, agent_path=('s',)), Evidence(step_id=2)]
    assert [(e.agent_path, e.step_id) for e in result('x', 'A', 1, ev).evidence] == [((), 2), ((), 3), (('s',), 1)]


def test_source_coverage_failure_keeps_signal_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    coverage = Precondition(name='source tool activity represented', met=False, detail='unrepresented: shell @step 2')
    monkeypatch.setattr(registry.pre, 'source_tool_coverage', lambda _ctx: coverage)
    evidence = [Evidence(step_id=2, call_id='c1', reason='observed call')]
    computed = SignalResult(
        name='tool_call_count', group='B', value=1, evidence=evidence,
        reason='one call observed', rule_version='rules-7',
    )

    failed = registry._require_source_tool_coverage(computed, context(traj([user()])))

    assert failed.value is None
    assert failed.no_basis == coverage.detail
    assert failed.evidence == evidence
    assert failed.reason == 'one call observed'
    assert failed.rule_version == 'rules-7'
    assert failed.preconditions[-1] == coverage


def test_inherited_coverage_failure_keeps_tag_diagnostics() -> None:
    evidence = [Evidence(step_id=3, reason='tagged')]
    computed = SignalResult(
        name='tool_churn', group='D', value=True, evidence=evidence,
        reason='retry condition met', rule_version='rules-8',
    )
    prior = {
        'tool_call_count': SignalResult(
            name='tool_call_count', group='B', no_basis='unrepresented source tool',
            preconditions=[Precondition(name='source tool activity represented', met=False, detail='missing result')],
        ),
    }

    failed = registry._inherit_source_tool_coverage(computed, 'tool_churn', prior)

    assert failed.value is None
    assert failed.no_basis == 'tool-dependent metrics unavailable: tool_call_count'
    assert failed.evidence == evidence
    assert failed.reason == 'retry condition met'
    assert failed.rule_version == 'rules-8'
    assert failed.preconditions[-1].name == 'source tool activity represented'
    assert failed.preconditions[-1].met is False


def test_report_values_omit_no_basis_but_keep_zero() -> None:
    report = SignalReport(
        trajectory_id=None,
        results={
            'zero': result('zero', 'A', 0),
            'none': SignalResult(name='none', group='B', no_basis='n/a'),
        },
        config_version='v',
    )
    assert report.values() == {'zero': 0}


def test_every_signal_shares_one_context_and_sees_earlier_results(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def first(ctx):
        seen.append(ctx)
        return result('first', 'A', 1)

    def second(ctx):
        seen.append(ctx)
        return result('second', 'D', ctx.results['first'].value + 1)

    monkeypatch.setattr(registry, 'SIGNALS', {'first': ('A', first), 'second': ('D', second)})
    root = traj([user(), agent(calls=[call('A')], results=[ok()])])
    report = compute_signals(root)
    assert report.values() == {'first': 1, 'second': 2}
    assert seen[0] is seen[1]
    assert seen[0].trajectory is root
    assert len(seen[0].walked) == 2 and len(seen[0].calls) == 1


def test_a_failing_walk_still_returns_a_report_with_every_signal_no_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    _table(monkeypatch)

    def broken(trajectory, config):
        msg = 'cannot walk'
        raise RuntimeError(msg)

    monkeypatch.setattr(registry.SignalContext, 'build', broken)
    report = compute_signals(traj([user()], trajectory_id='t9'))
    assert report.trajectory_id == 't9'
    assert {n: r.no_basis for n, r in report.results.items()} == {
        'fine': 'walk failed: RuntimeError: cannot walk',
        'boom': 'walk failed: RuntimeError: cannot walk',
    }
    assert [r.group for r in report.results.values()] == ['A', 'B']
    assert list(compute_signals(traj([user()]), only=['fine']).results) == ['fine']
    with pytest.raises(ValueError, match='nope'):
        compute_signals(traj([user()]), only=['nope'])
