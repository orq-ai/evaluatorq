"""Group B signal behavior on ATIF tool calls and observations."""

from __future__ import annotations

from evaluatorq.signals import compute_signals

from tests.signals.conftest import agent, call, failed, ok, traj


def _values(trajectory, *names):
    return compute_signals(trajectory, only=names).results


def test_error_retry_and_success_evidence():
    failed_call = call('fetch', {'q': 'old'}, 'failed')
    retry_call = call('fetch', {'q': 'new'}, 'retry')
    trajectory = traj([
        agent(calls=[failed_call], results=[failed('Error: offline')]),
        agent(calls=[retry_call], results=[ok('fresh')]),
    ])

    values = _values(trajectory, 'tool_error_count', 'tool_retry_count', 'tool_succeeded_after_retry_count')
    assert values['tool_error_count'].value == 1
    assert values['tool_error_count'].evidence[0].call_id == 'failed'
    assert values['tool_retry_count'].value == 1
    assert values['tool_retry_count'].evidence[0].related_call_ids == ['failed']
    assert values['tool_succeeded_after_retry_count'].value == 1
    assert values['tool_succeeded_after_retry_count'].evidence[0].related_call_ids == ['failed']


def test_duplicate_loops_and_result_sizes():
    repeated = [call('lookup', {'q': 'same'}, f'c{i}') for i in range(3)]
    trajectory = traj([
        agent(calls=[repeated[0]], results=[ok('é')]),
        agent(calls=[repeated[1]], results=[ok('{}')]),
        agent(calls=[repeated[2]], results=[ok('done')]),
    ])

    values = _values(
        trajectory,
        'duplicate_tool_call_count',
        'identical_tool_call_run_count',
        'tool_loop_count',
        'distinct_tool_arg_ratio',
        'empty_tool_result_count',
        'max_tool_result_bytes',
        'total_tool_result_bytes',
        'consecutive_same_tool_max',
    )
    assert values['duplicate_tool_call_count'].value == 2
    assert values['identical_tool_call_run_count'].value == 1
    assert values['tool_loop_count'].value == 1
    assert values['distinct_tool_arg_ratio'].value == 0.3333
    assert values['empty_tool_result_count'].value == 1
    assert values['max_tool_result_bytes'].value == 4
    assert values['total_tool_result_bytes'].value == 8
    assert values['consecutive_same_tool_max'].value == 3


def test_schema_violation_and_missing_schema_jsonschema_basis():
    schema = {'type': 'object', 'required': ['query'], 'properties': {'query': {'type': 'string'}}}
    trajectory = traj(
        [agent(calls=[call('search', {'wrong': 1}, 'bad')], results=[ok()])],
        tool_definitions=[{'type': 'function', 'function': {'name': 'search', 'parameters': schema}}],
    )
    report = _values(trajectory, 'invalid_schema_tool_call_count')['invalid_schema_tool_call_count']
    assert report.value == 1
    assert report.evidence[0].call_id == 'bad'
    assert 'required property' in report.evidence[0].reason

    missing = traj([agent(calls=[call('search', {})], results=[ok()])])
    no_schema = _values(missing, 'invalid_schema_tool_call_count')['invalid_schema_tool_call_count']
    assert no_schema.no_basis is not None
    assert no_schema.value is None


def test_schema_validation_reports_no_basis_without_optional_dependency(monkeypatch):
    import builtins

    schema = {'type': 'object', 'required': ['query']}
    trajectory = traj(
        [agent(calls=[call('search', {}, 'call')], results=[ok()])],
        tool_definitions=[{'type': 'function', 'function': {'name': 'search', 'parameters': schema}}],
    )
    original_import = builtins.__import__

    def without_jsonschema(name, *args, **kwargs):
        if name == 'jsonschema' or name.startswith('jsonschema.'):
            raise ImportError('optional dependency unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_jsonschema)
    report = _values(trajectory, 'invalid_schema_tool_call_count')['invalid_schema_tool_call_count']
    assert report.value is None
    assert report.no_basis == 'jsonschema is unavailable'


def test_command_family_runs_and_oscillation():
    trajectory = traj([
        agent(calls=[call('Bash', {'command': 'git status'}, 'a')], results=[ok()]),
        agent(calls=[call('Bash', {'command': 'orq traces'}, 'b')], results=[ok()]),
        agent(calls=[call('Bash', {'command': 'git status'}, 'c')], results=[ok()]),
        agent(calls=[call('Bash', {'command': 'orq traces'}, 'd')], results=[ok()]),
    ])
    values = _values(trajectory, 'consecutive_command_family_max', 'tool_oscillation_count', 'tool_loop_count')
    assert values['consecutive_command_family_max'].value == 1
    assert values['tool_oscillation_count'].value == 1
    assert values['tool_loop_count'].value == 1
