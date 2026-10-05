"""Group B signal behavior on ATIF tool calls and observations."""

from __future__ import annotations

from evaluatorq.formats.atif import AtifObservationResult
from evaluatorq.signals import compute_signals
from evaluatorq.signals.config import SignalsConfig

from tests.signals.conftest import agent, bare, call, failed, ok, traj


def _values(trajectory, *names, config=None):
    return compute_signals(trajectory, config=config, only=names).results


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


def test_raw_argument_provenance_keeps_distinct_failed_parses_apart():
    def raw_call(text: str, call_id: str):
        return call('lookup', {}, call_id).model_copy(update={'extra': {'evaluatorq.raw_arguments': text}})

    trajectory = traj([
        agent(calls=[raw_call('{bad-one', 'a')]),
        agent(calls=[raw_call('{bad-two', 'b')]),
        agent(calls=[raw_call('{bad-one', 'c')]),
    ])

    duplicate = _values(trajectory, 'duplicate_tool_call_count')['duplicate_tool_call_count']
    assert duplicate.value == 1
    assert duplicate.evidence[0].call_id == 'c'


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


def test_response_local_schema_is_selected_at_each_tool_call():
    from openai.types.responses import Response

    from evaluatorq.formats.responses import ResponsesConversation

    items = [
        {'type': 'function_call', 'call_id': 'a', 'name': 'lookup', 'arguments': '{"wrong": 1}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'one'},
        {'type': 'function_call', 'call_id': 'b', 'name': 'lookup', 'arguments': '{"b": 1}'},
    ]
    first_tools = [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object', 'required': ['a']}}]
    second_tools = [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object', 'required': ['b']}}]
    base = {
        'created_at': 1, 'model': 'gpt-x', 'object': 'response', 'parallel_tool_calls': False,
        'tool_choice': 'auto', 'status': 'completed', 'usage': None,
    }
    conversation = ResponsesConversation(items=items, responses=[
        Response.model_validate({**base, 'id': 'resp_a', 'output': [items[0]], 'tools': first_tools}),
        Response.model_validate({**base, 'id': 'resp_b', 'output': [items[2]], 'tools': second_tools}),
    ])
    trajectory = conversation.to_atif()

    signal = _values(trajectory, 'invalid_schema_tool_call_count')['invalid_schema_tool_call_count']
    assert signal.value == 1
    assert [item.call_id for item in signal.evidence] == ['a']


def test_unmapped_responses_tool_activity_invalidates_only_tool_dependent_signals():
    from evaluatorq.formats.responses import ResponsesConversation

    trajectory = ResponsesConversation(items=[
        {'type': 'custom_tool_call', 'call_id': 'custom-1', 'name': 'lookup', 'input': 'query'},
        {'type': 'custom_tool_call_output', 'call_id': 'custom-1', 'output': 'answer'},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'finished'}]},
    ]).to_atif()
    values = _values(
        trajectory,
        'tool_call_count',
        'llm_call_count',
        'max_autonomous_steps',
        'terminal_answer_present',
        'tool_churn',
    )

    assert values['tool_call_count'].value is None
    assert values['tool_call_count'].preconditions[-1].name == 'source tool activity represented'
    assert values['llm_call_count'].value == 2
    assert values['max_autonomous_steps'].value is None
    assert values['terminal_answer_present'].value is None
    assert values['tool_churn'].value is None


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


def test_unusable_tool_schema_has_no_basis():
    trajectory = traj(
        [agent(calls=[call('search', {'query': 'x'}, 'call')], results=[ok()])],
        tool_definitions=[{
            'type': 'function',
            'function': {'name': 'search', 'parameters': {'type': 'not-a-jsonschema-type'}},
        }],
    )

    report = _values(trajectory, 'invalid_schema_tool_call_count')['invalid_schema_tool_call_count']
    assert report.value is None
    assert 'unusable tool schema' in (report.no_basis or '')
    assert any(pc.name == 'tool schemas valid' and pc.met is False for pc in report.preconditions)


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
    assert values['tool_oscillation_count'].evidence[0].reason == '4-call oscillation: git status ↔ orq traces'


def test_command_family_max_evidence_contains_only_longest_run():
    trajectory = traj([
        agent(calls=[call('Bash', {'command': command}, str(index))])
        for index, command in enumerate([
            'git status',
            'git diff',
            'orq traces',
            'orq traces list',
            'orq traces get',
        ])
    ])

    report = _values(trajectory, 'consecutive_command_family_max')['consecutive_command_family_max']

    assert report.value == 3
    assert len(report.evidence) == 1
    assert report.evidence[0].call_id == '2'
    assert report.evidence[0].reason == '3x orq traces'


def test_command_family_skips_cd_and_environment_prefixes():
    trajectory = traj([
        agent(calls=[call('Bash', {'command': 'cd /tmp && FOO=bar git status'}, 'first')]),
        agent(calls=[call('Bash', {'command': 'cd /tmp && FOO=bar orq traces'}, 'second')]),
    ])

    report = _values(trajectory, 'consecutive_command_family_max')['consecutive_command_family_max']
    assert report.value == 1


def test_oscillation_of_same_tool_with_different_arguments_is_labeled():
    trajectory = traj([
        agent(calls=[call('lookup', {'q': 'a'}, 'a1')], results=[ok()]),
        agent(calls=[call('lookup', {'q': 'b'}, 'b1')], results=[ok()]),
        agent(calls=[call('lookup', {'q': 'a'}, 'a2')], results=[ok()]),
        agent(calls=[call('lookup', {'q': 'b'}, 'b2')], results=[ok()]),
    ])
    oscillation = _values(trajectory, 'tool_oscillation_count')['tool_oscillation_count']
    assert oscillation.value == 1
    assert oscillation.evidence[0].reason == '4-call oscillation: lookup (two argument sets)'


def test_alternate_retry_definition_and_window():
    trajectory = traj([
        agent(calls=[call('fetch', {'q': 1}, 'failed')], results=[failed()]),
        agent(calls=[call('other', {}, 'other')], results=[ok()]),
        agent(calls=[call('fetch', {'q': 1}, 'retry')], results=[ok()]),
    ])
    within_two = SignalsConfig(retry_definition='same_tool_args_within_n', retry_window=2)
    within_one = SignalsConfig(retry_definition='same_tool_args_within_n', retry_window=1)

    wide = _values(trajectory, 'tool_retry_count', 'tool_succeeded_after_retry_count', config=within_two)
    narrow = _values(trajectory, 'tool_retry_count', config=within_one)
    assert wide['tool_retry_count'].value == 1
    assert wide['tool_retry_count'].evidence[0].reason == ''
    assert wide['tool_succeeded_after_retry_count'].value == 1
    assert narrow['tool_retry_count'].value == 0


def test_exact_canonicalisation_preserves_argument_key_order():
    trajectory = traj([
        agent(calls=[call('lookup', {'a': 1, 'b': 2}, 'first')], results=[ok()]),
        agent(calls=[call('lookup', {'b': 2, 'a': 1}, 'second')], results=[ok()]),
    ])
    sorted_result = _values(trajectory, 'duplicate_tool_call_count')['duplicate_tool_call_count']
    exact_config = SignalsConfig(canonicalisation='exact')
    exact_result = _values(trajectory, 'duplicate_tool_call_count', config=exact_config)['duplicate_tool_call_count']
    assert sorted_result.value == 1
    assert exact_result.value == 0


def test_status_and_content_detection_and_unknown_status_basis():
    trajectory = traj([agent(calls=[call('run', {})], results=[bare('Traceback: failed')])])
    status_only = _values(trajectory, 'tool_error_count')['tool_error_count']
    sniff = _values(
        trajectory,
        'tool_error_count',
        config=SignalsConfig(error_detection='status_and_content'),
    )['tool_error_count']
    assert status_only.value is None
    assert status_only.no_basis == 'explicit error status: 0 of 1 tool results carry an error status'
    assert sniff.value == 1
    assert sniff.evidence[0].reason == "matched 'Traceback'"
    assert sniff.preconditions[0].met is False
    assert sniff.preconditions[0].required is False


def test_empty_values_and_literals_are_configurable():
    trajectory = traj([
        agent(calls=[call('run', {}, 'blank')], results=[ok('')]),
        agent(calls=[call('run', {}, 'literal')], results=[ok('(no output)')]),
        agent(calls=[call('run', {}, 'array')], results=[ok('[]')]),
    ])
    defaults = _values(trajectory, 'empty_tool_result_count')['empty_tool_result_count']
    config = SignalsConfig(empty_values=frozenset({'[]'}), empty_literals=('(no output)',))
    customized = _values(trajectory, 'empty_tool_result_count', config=config)['empty_tool_result_count']
    assert defaults.value == 2
    assert customized.value == 2
    assert [item.reason for item in customized.evidence] == ["empty: '(no output)'", "empty: '[]'"]


def test_empty_content_list_is_distinct_from_empty_text():
    trajectory = traj([
        agent(
            calls=[call('run', {}, 'empty-list')], results=[AtifObservationResult(content=[], extra={'status': 'ok'})]
        ),
        agent(calls=[call('run', {}, 'empty-text')], results=[ok('')]),
    ])
    config = SignalsConfig(empty_values=frozenset({'[]'}))
    empty = _values(trajectory, 'empty_tool_result_count', config=config)['empty_tool_result_count']
    assert empty.value == 1
    assert empty.evidence[0].call_id == 'empty-list'
    assert empty.evidence[0].reason == "empty: '[]'"
