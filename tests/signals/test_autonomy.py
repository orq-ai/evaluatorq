"""Group C autonomy, delegation and timing signals."""

from __future__ import annotations

import json
from pathlib import Path

from evaluatorq.formats.otel import OtelTrace
from evaluatorq.signals import compute_signals

from .conftest import agent, call, ok, spawned, traj, user

FIXTURES = Path(__file__).parent.parent / 'formats' / 'fixtures'


def _run(trajectory, *names):
    return compute_signals(trajectory, only=names).results


def _timed_user(timestamp: str):
    spec = user()
    spec.kwargs['timestamp'] = timestamp
    return spec


def test_autonomous_segments_count_llm_responses_and_tool_calls() -> None:
    trajectory = traj([
        user('first'),
        agent(calls=[call('one', call_id='one'), call('two', call_id='two')], results=[ok(), ok()]),
        agent('answer'),
        user('second'),
        agent(calls=[call('three')], results=[ok()]),
    ])
    values = _run(trajectory, 'autonomous_segment_count', 'max_autonomous_steps', 'avg_autonomous_steps')
    assert values['autonomous_segment_count'].value == 2
    assert values['max_autonomous_steps'].value == 4
    assert values['avg_autonomous_steps'].value == 3.0
    assert values['max_autonomous_steps'].evidence[0].step_id == 1


def test_cycle_runs_are_isolated_by_subagent_path() -> None:
    child = traj([agent(calls=[call('child-1')]), agent(calls=[call('child-2')])], trajectory_id='child')
    root = traj(
        [
            agent(calls=[call('root-1')], results=[spawned('child')]),
            agent(calls=[call('root-2')]),
        ],
        subagents=[child],
    )
    cycles = _run(root, 'max_llm_tool_cycles')['max_llm_tool_cycles']
    assert cycles.value == 2
    assert cycles.evidence[0].agent_path in ((), ('child',))
    assert len(cycles.evidence[0].related) == 1


def test_delegation_and_terminal_answer_use_agent_path_evidence() -> None:
    child = traj([agent('child result')], trajectory_id='child')
    root = traj(
        [user(), agent(calls=[call('delegate')], results=[spawned('child')]), agent('done')],
        subagents=[child],
    )
    values = _run(root, 'subagent_step_share', 'subagent_message_share', 'terminal_answer_present')
    assert values['subagent_step_share'].value == 0.25
    assert values['subagent_message_share'].value == 0.25
    assert values['subagent_step_share'].evidence[0].agent_path == ('child',)
    assert values['terminal_answer_present'].value is True
    assert values['terminal_answer_present'].evidence[0].agent_path == ()


def test_parallel_batches_and_human_interruptions() -> None:
    trajectory = traj([
        user(),
        agent(calls=[call('a', call_id='a'), call('b', call_id='b')]),
        user('continue'),
        agent('finished'),
    ])
    values = _run(trajectory, 'parallel_tool_batch_count', 'max_parallel_tool_calls', 'human_interruption_count')
    assert values['parallel_tool_batch_count'].value == 1
    assert values['parallel_tool_batch_count'].evidence[0].related_call_ids == ['a', 'b']
    assert values['max_parallel_tool_calls'].value == 2
    interruption = values['human_interruption_count']
    assert interruption.value == 1
    assert interruption.evidence[0].related == [((), 2)]
    assert interruption.evidence[0].related_call_ids == ['a', 'b']


def test_real_orq_fixture_provides_exact_timing() -> None:
    raw = json.loads((FIXTURES / 'otel' / 'orq_agent_subagent.json').read_text())
    trajectory = OtelTrace.from_orq(raw).to_atif()
    values = _run(trajectory, 'wall_time_ms', 'active_time_ms', 'llm_time_ms', 'tool_time_ms')
    for name, signal in values.items():
        assert signal.value is not None, name
        assert signal.approximate is False, name
    assert values['tool_time_ms'].evidence[0].call_id == 'call_1'


def test_wall_and_segment_timing_track_exact_bounds_and_iso_extrema_separately() -> None:
    start = 1767225600.0
    trajectory = traj([
        _timed_user('2026-01-01T00:00:00Z'),
        agent(
            calls=[call('tool', call_id='tool-call')],
            results=[ok(start_timestamp=start + 3, end_timestamp=start + 5)],
            timestamp='2026-01-01T00:00:02Z',
            extra={'invocation': {'start_timestamp': start + 2, 'end_timestamp': start + 3}},
        ),
        agent(
            'answer',
            timestamp='2026-01-01T00:00:06Z',
            extra={'invocation': {'start_timestamp': start + 6, 'end_timestamp': start + 7}},
        ),
    ])
    values = _run(trajectory, 'wall_time_ms', 'active_time_ms', 'max_autonomous_duration_ms')
    assert values['wall_time_ms'].value == 7000
    assert values['wall_time_ms'].approximate is True
    assert values['active_time_ms'].value == 4000
    assert values['active_time_ms'].approximate is False
    duration = values['max_autonomous_duration_ms']
    assert duration.value == 7000
    assert duration.approximate is False
    assert duration.evidence[0].related_call_ids == ['tool-call']


def test_iso_timestamps_use_approximate_fallback_intervals() -> None:
    trajectory = traj([
        _timed_user('2026-01-01T00:00:00Z'),
        agent(
            calls=[call('tool', call_id='c')],
            results=[ok()],
            timestamp='2026-01-01T00:00:02Z',
        ),
        agent('answer', timestamp='2026-01-01T00:00:05Z'),
    ])
    values = _run(trajectory, 'wall_time_ms', 'active_time_ms', 'llm_time_ms', 'tool_time_ms')
    assert values['wall_time_ms'].value == 5000
    assert values['wall_time_ms'].approximate is True
    assert values['llm_time_ms'].value == 5000
    assert values['tool_time_ms'].value == 0
    assert values['active_time_ms'].value == 5000
    assert all(signal.approximate for signal in values.values())


def test_timing_without_timestamps_has_no_basis() -> None:
    values = _run(traj([user(), agent('answer')]), 'wall_time_ms', 'active_time_ms', 'llm_time_ms', 'tool_time_ms')
    assert all(signal.value is None and signal.no_basis is not None for signal in values.values())


def test_max_autonomous_duration_uses_the_longest_timed_root_segment() -> None:
    trajectory = traj([
        _timed_user('2026-01-01T00:00:00Z'),
        agent('a', timestamp='2026-01-01T00:00:01Z'),
        _timed_user('2026-01-01T00:00:03Z'),
        agent('b', timestamp='2026-01-01T00:00:08Z'),
    ])
    signal = _run(trajectory, 'max_autonomous_duration_ms')['max_autonomous_duration_ms']
    assert signal.value == 5000
    assert signal.approximate is True
    assert signal.evidence[0].step_id == 3
