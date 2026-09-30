"""Registry: failure isolation, `only` handling, and the report's no-basis rule."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluatorq.signals import registry
from evaluatorq.signals.models import Evidence, Precondition, SignalReport, SignalResult, result
from evaluatorq.signals.registry import compute_signals

from .conftest import agent, call, ok, traj, user


def _fine(ctx):
    return result('fine', 'A', 3)


def _boom(ctx):
    msg = 'bad rule'
    raise RuntimeError(msg)


def _table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, 'SIGNALS', {'fine': ('A', _fine), 'boom': ('B', _boom)})


def test_the_registry_starts_empty_and_reports_the_config_version() -> None:
    report = compute_signals(traj([user(), agent()], trajectory_id='t1'))
    assert report.results == {}
    assert report.trajectory_id == 't1'
    assert report.config_version == 'v3-local-cc-2026-09-29'


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
