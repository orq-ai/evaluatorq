"""Walker order, skipped steps, call/result pairing and the error tri-state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluatorq.formats.atif import AtifTrajectory
from evaluatorq.formats.otel import OtelTrace
from evaluatorq.signals.walk import calls, finish_reasons, infer_provider, is_llm_step, llm_times, walk

from .conftest import agent, bare, call, compaction, copied, failed, ok, spawned, system, traj, user

FIXTURES = Path(__file__).parent.parent / 'formats' / 'fixtures'


def _ids(walked):
    return [(w.agent_path, w.step.step_id, w.depth) for w in walked]


def test_walk_places_the_subagent_after_the_referencing_step() -> None:
    child = traj([user('task'), agent('child answer')], trajectory_id='child')
    root = traj(
        [user(), agent(calls=[call('delegate')], results=[spawned('child')]), agent('final')], subagents=[child]
    )
    walked = walk(root)
    assert _ids(walked) == [((), 1, 0), ((), 2, 0), (('child',), 1, 1), (('child',), 2, 1), ((), 3, 0)]
    assert not any(w.unlinked for w in walked)


def test_walk_nests_subagents_of_subagents() -> None:
    grand = traj([agent('g')], trajectory_id='grand')
    child = traj([agent(calls=[call('d')], results=[spawned('grand')])], trajectory_id='child', subagents=[grand])
    root = traj([agent(calls=[call('d')], results=[spawned('child')])], subagents=[child])
    assert _ids(walk(root)) == [((), 1, 0), (('child',), 1, 1), (('child', 'grand'), 1, 2)]


def test_walk_appends_unreferenced_subagents_flagged_at_depth_one() -> None:
    stray = traj([agent('lost')], trajectory_id='stray')
    root = traj([user(), agent('a')], subagents=[stray])
    walked = walk(root)
    assert _ids(walked) == [((), 1, 0), ((), 2, 0), (('stray',), 1, 1)]
    assert [w.unlinked for w in walked] == [False, False, True]


def test_walk_ignores_a_reference_to_a_subagent_that_is_not_embedded() -> None:
    root = traj([agent(calls=[call('d')], results=[spawned('missing')])])
    assert _ids(walk(root)) == [((), 1, 0)]


def test_walk_places_a_subagent_once_when_referenced_twice() -> None:
    child = traj([agent('c')], trajectory_id='child')
    root = traj(
        [
            agent(calls=[call('d', call_id='c1')], results=[spawned('child', call_id='c1')]),
            agent(calls=[call('d', call_id='c2')], results=[spawned('child', call_id='c2')]),
        ],
        subagents=[child],
    )
    assert _ids(walk(root)) == [((), 1, 0), (('child',), 1, 1), ((), 2, 0)]


def test_walk_skips_compaction_and_copied_context_steps() -> None:
    root = traj([user('a'), copied(), compaction(), agent('b')])
    assert [w.step.step_id for w in walk(root)] == [1, 4]


def test_walk_phoenix_fixture_orders_the_embedded_subagent_and_skips_compaction_and_copied_steps() -> None:
    doc = json.loads((FIXTURES / 'atif' / 'phoenix_v17_embedded_subagents.json').read_text())
    walked = walk(AtifTrajectory.from_json(doc))
    assert _ids(walked) == [
        ((), 1, 0),
        ((), 2, 0),
        (('child-doc',), 2, 1),
        ((), 4, 0),
    ]


def test_walk_otel_fixture_nests_the_invoke_agent_subagent() -> None:
    trace = OtelTrace.from_orq(json.loads((FIXTURES / 'otel' / 'orq_agent_subagent.json').read_text()))
    root = trace.to_atif()
    assert root.subagent_trajectories
    walked = walk(root)
    assert len(walked) == len(root.steps) + sum(len(s.steps) for s in root.subagent_trajectories)
    sub_id = root.subagent_trajectories[0].trajectory_id
    assert any(w.agent_path == (sub_id,) and w.depth == 1 and not w.unlinked for w in walked)
    first_sub = next(i for i, w in enumerate(walked) if w.agent_path)
    assert walked[first_sub - 1].agent_path == ()
    assert walked[first_sub - 1].step.observation is not None


def test_calls_pair_results_by_source_call_id_within_the_step() -> None:
    root = traj([
        agent(
            calls=[call('A', call_id='a'), call('B', call_id='b')],
            results=[ok('for b', call_id='b')],
        ),
        agent(calls=[call('A', call_id='a')], results=[ok('second a', call_id='a')]),
    ])
    records = calls(walk(root))
    assert [(r.seq, r.call.function_name, r.result and r.result.content) for r in records] == [
        (0, 'A', None),
        (1, 'B', 'for b'),
        (2, 'A', 'second a'),
    ]
    assert records[2].step.step.step_id == 2


def test_calls_keep_the_first_result_for_a_call_id() -> None:
    root = traj([agent(calls=[call('A', call_id='a')], results=[ok('first', call_id='a'), ok('later', call_id='a')])])
    assert [r.result.content for r in calls(walk(root)) if r.result] == ['first']


@pytest.mark.parametrize(
    ('result', 'expected'),
    [
        (failed(), True),
        (ok(), False),
        (bare(), None),
        (ok(error_type='Timeout'), True),
        (bare().model_copy(update={'extra': {'error_type': 'Boom'}}), True),
        (bare().model_copy(update={'extra': {'other': 1}}), None),
    ],
)
def test_is_error_is_a_tri_state(result, expected) -> None:
    root = traj([agent(calls=[call('A')], results=[result])])
    assert calls(walk(root))[0].is_error is expected


def test_is_error_is_unknown_for_an_orphan_call() -> None:
    assert calls(walk(traj([agent(calls=[call('A')])])))[0].is_error is None


def test_llm_step_and_model_fall_back_to_the_trajectory_agent() -> None:
    root = traj([user(), agent('a'), agent('b', model='openai/gpt-x'), agent('c', llm_call_count=0)], model='claude-x')
    walked = walk(root)
    assert [is_llm_step(w) for w in walked] == [False, True, True, False]


def test_llm_step_needs_a_model_or_call_count() -> None:
    assert [is_llm_step(w) for w in walk(traj([agent('a')]))] == [False]
    assert [is_llm_step(w) for w in walk(traj([agent('a', llm_call_count=2)]))] == [True]


@pytest.mark.parametrize(
    ('model', 'provider'),
    [('openai/gpt-5', 'openai'), ('claude-sonnet-5', 'anthropic'), ('mystery', None), (None, None)],
)
def test_infer_provider(model, provider) -> None:
    assert infer_provider(model) == provider


def test_llm_times_prefer_invocation_then_iso() -> None:
    inv = {'invocation': {'start_timestamp': 10.0, 'end_timestamp': 12.5, 'finish_reasons': ['length', 'stop']}}
    root = traj([agent('a', extra=inv), agent('b', timestamp='2026-01-01T00:00:00Z'), agent('c')])
    a, b, c = (llm_times(w) for w in walk(root))
    assert (a.start, a.end, a.approximate) == (10.0, 12.5, False)
    assert b.start == 1767225600.0 and b.end is None and b.approximate
    assert (c.start, c.end) == (None, None)
    assert finish_reasons(walk(root)[0].step) == ['length', 'stop']
