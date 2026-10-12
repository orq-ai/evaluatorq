"""Unit tests for `evaluatorq.insights.transcript` — the classifier's text views and tool counts."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import pytest
from evaluatorq.common.model_input import serialized_question_chars
from evaluatorq.common.trace_document import ensure_trace_document

from evaluatorq.insights.readable_style import ReadableStyle
from evaluatorq.insights.transcript import (
    first_command,
    full_conversation_view,
    full_conversation_view_with_message_spans,
    is_risky_command,
    tool_activity_chunks,
    tool_inventory,
    tool_stats,
)
from evaluatorq.trace_finder.models import TraceRecord


def _call(call_id: str, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': raw}}


def _trace(*messages: dict[str, Any]) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='t1',
        span_id='s1',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=messages,
        project='default',
        model='claude',
        provider='anthropic',
        status='completed',
        product='agent',
        trace_type='conversation',
    )


CODING_TRACE = _trace(
    {'role': 'user', 'content': 'Fix the failing parser test.<system-reminder>injected context</system-reminder>'},
    {
        'role': 'assistant',
        'content': 'Looking at the parser.',
        'tool_calls': [
            _call('c1', 'Skill', {'skill': 'systematic-debugging'}),
            _call('c2', 'Bash', {'command': 'cd repo && uv run pytest tests/test_parser.py -q'}),
            _call('c3', 'Edit', {'file_path': 'src/parser.py', 'old_string': 'a', 'new_string': 'b'}),
            _call('c4', 'Bash', {'command': 'git push --force origin main'}),
        ],
    },
    {'role': 'tool', 'tool_call_id': 'c1', 'content': 'loaded'},
    {'role': 'tool', 'tool_call_id': 'c2', 'content': 'Error: 1 failed, 3 passed'},
    {'role': 'tool', 'tool_call_id': 'c3', 'content': 'ok'},
    {'role': 'tool', 'tool_call_id': 'c4', 'content': 'forced update'},
    {'role': 'assistant', 'content': 'Fixed and pushed.'},
)


@pytest.mark.parametrize(
    ('command', 'expected'),
    [
        ('cd x && FOO=1 uv run pytest -q', 'pytest'),
        ('gh pr create --title x', 'gh pr'),
        ('rtk git status', 'git status'),
        ('env -u ORQ_API_KEY orq traces search --limit 3', 'orq traces'),
        ('timeout 60 npm test', 'npm test'),
        ('D=$(mktemp -d)\necho "$D"', 'echo'),
        ('for f in *.py; do ruff check "$f"; done', 'ruff'),
        ('ls -la \\\n  src', 'ls'),
        ('cd somewhere', '(none)'),
    ],
)
def test_first_command_names_the_program(command: str, expected: str) -> None:
    assert first_command(command) == expected


def test_risky_commands_are_recognised() -> None:
    assert is_risky_command('git push origin main')
    assert is_risky_command('rm -rf build')
    assert is_risky_command('gh pr merge 12 --squash')
    assert not is_risky_command('git status')
    assert not is_risky_command('uv run pytest -q')


def test_tool_stats_count_tools_skills_and_commands_from_messages() -> None:
    stats = tool_stats(CODING_TRACE)

    assert stats.tools == {'Bash': 2, 'Edit': 1}
    assert stats.skills == {'systematic-debugging': 1}
    assert stats.commands == {'pytest': 1, 'git push': 1}


def test_tool_stats_read_a_skill_from_arguments_cut_mid_string() -> None:
    trace = _trace(
        {'role': 'user', 'content': 'go'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call('c1', 'Skill', '{"skill": "hate", "args": "a very lo')],
        },
    )

    assert tool_stats(trace).skills == {'hate': 1}



def test_full_conversation_view_collapses_tools_by_default_and_ignores_tool_max_chars() -> None:
    expected = (
        '0 user: Fix the failing parser test.[harness reminder omitted]\n'
        '1 assistant: Looking at the parser.\n'
        '1 tools: Skill: systematic-debugging, Bash: pytest, Edit: src/parser.py, Bash: `git push --force origin main`\n'
        '6 assistant: Fixed and pushed.'
    )

    assert full_conversation_view(CODING_TRACE, budget=10_000) == expected
    assert full_conversation_view(CODING_TRACE, budget=10_000, tool_max_chars=500) == expected


def test_full_conversation_view_per_call_lists_each_call_and_cuts_results() -> None:
    log = 'HEAD' + 'x' * 500 + 'TAIL'
    trace = _trace(
        {'role': 'user', 'content': 'Run the tests.'},
        {
            'role': 'assistant',
            'content': 'Running.',
            'tool_calls': [
                _call('c1', 'Bash', {'command': 'uv run pytest -q\necho done'}),
                _call('c2', 'Read', {'file_path': 'src/a.py'}),
                _call('c3', 'Read', {'file_path': 'src/b.py'}),
            ],
        },
        {'role': 'tool', 'tool_call_id': 'c1', 'content': log},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'line one\nline two'},
        {'role': 'assistant', 'content': 'Done.'},
    )

    lines = full_conversation_view(trace, budget=10_000, tools='per_call', tool_max_chars=40).splitlines()

    assert lines[2] == '1 tool input Bash: uv run pytest -q ⏎ echo done'
    assert lines[3] == '1 tool input Read: {"file_path": "src/a.py"}'
    assert lines[4] == '1 tool input Read [pending]: {"file_path": "src/b.py"}'
    assert lines[5].startswith('2 tool result c1: HEAD')
    assert lines[5].endswith('TAIL')
    assert '[... 468 chars ...]' in lines[5]
    assert lines[6:8] == ['3 tool result c2: line one', 'line two']
    assert lines[8] == '4 assistant: Done.'
    # 0 (the default) keeps the per-call lines but leaves every result out
    without_output = full_conversation_view(trace, budget=10_000, tools='per_call')
    assert 'HEAD' not in without_output
    assert 'tool result' not in without_output
    assert '1 tool input Read [pending]: {"file_path": "src/b.py"}' in without_output


def test_full_conversation_view_per_call_marks_errors() -> None:
    view = full_conversation_view(CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=200)

    assert '1 tool input Bash [error]: cd repo && uv run pytest tests/test_parser.py -q' in view
    assert '3 tool result c2: Error: 1 failed, 3 passed' in view
    assert '1 tool input Bash: git push --force origin main' in view
    assert 'injected context' not in view


def test_full_conversation_view_caps_each_message_type_and_minus_one_keeps_it_whole() -> None:
    long_user = 'U' * 100 + 'u' * 20_000 + 'END'
    long_reply = 'A' * 100 + 'a' * 5_000 + 'Z' * 100
    trace = _trace(
        {'role': 'system', 'content': 'S' * 3_000},
        {'role': 'user', 'content': long_user},
        {'role': 'assistant', 'content': long_reply, 'tool_calls': [_call('c1', 'Read', {'file_path': 'p' * 900})]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'r' * 900},
    )

    view = full_conversation_view(trace, budget=100_000, tools='per_call', tool_max_chars=100)
    lines = view.splitlines()
    assert lines[0] == '0 system: ' + 'S' * 3_000  # system is whole by default
    assert lines[1].startswith('1 user: UUU') and lines[1].endswith('END')
    assert len(lines[1]) < 8_100
    assert lines[2] == '2 assistant: ' + long_reply[:500] + ' [... 4200 chars ...] ' + long_reply[-500:]
    assert len(lines[3]) == len('2 tool input Read: ') + 300
    assert '[... 800 chars ...]' in lines[4]

    whole = full_conversation_view(
        trace,
        budget=100_000,
        tools='per_call',
        user_chars=-1,
        assistant_edge=-1,
        tool_input_chars=-1,
        tool_max_chars=-1,
    )
    assert f'1 user: {long_user}' in whole
    assert f'2 assistant: {long_reply}' in whole
    assert 'p' * 900 in whole
    assert '3 tool result c1: ' + 'r' * 900 in whole
    assert 'chars ...]' not in whole


def test_full_conversation_view_flags_repeated_call_ids_by_their_own_result() -> None:
    trace = _trace(
        {'role': 'user', 'content': 'Run both.'},
        {
            'role': 'assistant',
            'tool_calls': [_call('dup', 'Bash', {'command': 'first'}), _call('dup', 'Bash', {'command': 'second'})],
        },
        {'role': 'tool', 'tool_call_id': 'dup', 'content': 'ok', 'is_error': True},
    )

    view = full_conversation_view(trace, budget=10_000, tools='per_call')

    assert '1 tool input Bash [error]: first' in view
    assert '1 tool input Bash [pending]: second' in view


def test_full_conversation_view_exclude_removes_types_from_the_default() -> None:
    view = full_conversation_view(CODING_TRACE, budget=10_000, exclude={'assistant', 'tool_input'})

    assert view == '0 user: Fix the failing parser test.[harness reminder omitted]'
    per_call = full_conversation_view(
        CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=-1, exclude={'tool_input', 'assistant'}
    )
    assert 'tool input' not in per_call
    assert '3 tool result c2: Error: 1 failed, 3 passed' in per_call


REASONING_TRACE = _trace(
    {'role': 'user', 'content': 'Why?'},
    {
        'role': 'assistant',
        'content': [{'type': 'thinking', 'thinking': 'PART-THOUGHT'}, {'type': 'text', 'text': 'Because.'}],
        'reasoning_content': 'FIELD-THOUGHT',
    },
    {
        'role': 'assistant',
        'content': [
            {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'SUMMARY-THOUGHT'}]},
            {'type': 'text', 'text': 'Done.'},
        ],
    },
)


def test_full_conversation_view_omits_reasoning_unless_included() -> None:
    assert 'THOUGHT' not in full_conversation_view(REASONING_TRACE, budget=10_000)

    view = full_conversation_view(REASONING_TRACE, budget=10_000, include={'user', 'assistant', 'reasoning'})

    assert view.splitlines() == [
        '0 user: Why?',
        '1 reasoning: FIELD-THOUGHT',
        '',
        'PART-THOUGHT',
        '1 assistant: Because.',
        '2 reasoning: SUMMARY-THOUGHT',
        '2 assistant: Done.',
    ]
    capped = full_conversation_view(
        _trace({'role': 'assistant', 'reasoning_content': 'R' * 100 + 'x' * 5_000 + 'END'}),
        budget=10_000,
        include={'reasoning'},
        reasoning_chars=200,
    )
    assert capped.startswith('0 reasoning: RRR')
    assert capped.endswith('END')
    assert '[... 4903 chars ...]' in capped


def test_full_conversation_view_reads_native_atif_reasoning() -> None:
    document = ensure_trace_document(_trace({'role': 'user', 'content': 'Why?'}, {'role': 'assistant', 'content': 'Because.'}))
    steps = list(document.trajectory.steps)
    steps[1] = steps[1].model_copy(update={'reasoning_content': 'NATIVE-THOUGHT'})
    document = document.model_copy(update={'trajectory': document.trajectory.model_copy(update={'steps': steps})})

    view = full_conversation_view(document, budget=10_000, include={'assistant', 'reasoning'})

    assert view.splitlines() == ['1 reasoning: NATIVE-THOUGHT', '1 assistant: Because.']
    assert 'NATIVE-THOUGHT' not in full_conversation_view(document, budget=10_000)


def test_full_conversation_view_include_filters_message_types() -> None:
    view = full_conversation_view(CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=-1, include={'user', 'tool_result'})

    assert view.splitlines() == [
        '0 user: Fix the failing parser test.[harness reminder omitted]',
        '2 tool result c1: loaded',
        '3 tool result c2: Error: 1 failed, 3 passed',
        '4 tool result c3: ok',
        '5 tool result c4: forced update',
    ]
    assert full_conversation_view(CODING_TRACE, budget=10_000, include={'assistant'}).splitlines() == [
        '1 assistant: Looking at the parser.',
        '6 assistant: Fixed and pushed.',
    ]


@pytest.mark.parametrize('padding', [0, 15, 30, 60])
def test_full_conversation_view_per_call_scrubs_secrets_in_inputs_and_outputs(padding: int) -> None:
    secret = 'ghp_' + 'aB3dE5gH7j' * 3
    trace = _trace(
        {'role': 'user', 'content': 'Deploy it.'},
        {
            'role': 'assistant',
            'tool_calls': [
                _call('c1', 'Bash', {'command': f'curl -H "Authorization: Bearer {secret}" https://api.example.com'}),
                _call('c2', 'Lookup', {'token': secret}),
                _call('c3', 'Bash', {'command': f'git push https://x:{secret}@github.com/o/r --force'}),
            ],
        },
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'a' * padding + f' {secret} ' + 'z' * 200},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'ok'},
    )

    per_call = full_conversation_view(trace, budget=10_000, tools='per_call', tool_max_chars=80)
    collapsed = full_conversation_view(trace, budget=10_000)

    for view in (per_call, collapsed):
        for size in (8, 12):
            assert secret[:size] not in view
            assert secret[-size:] not in view
    assert '1 tool input Bash: curl -H' in per_call
    assert '1 tool input Lookup: {"token":' in per_call


def test_full_conversation_view_over_budget_keeps_both_ends() -> None:
    calls = [_call(f'c{index}', 'Bash', {'command': f'step {index}'}) for index in range(60)]
    trace = _trace(
        {'role': 'user', 'content': 'Run every step.'},
        {'role': 'assistant', 'tool_calls': calls},
        *({'role': 'tool', 'tool_call_id': f'c{index}', 'content': f'output {index}'} for index in range(60)),
    )

    view = full_conversation_view(trace, budget=1_200, tools='per_call', tool_max_chars=40)

    assert len(view) <= 1_200
    assert view.startswith('0 user: Run every step.')
    assert 'chars left out ...]' in view
    assert view.endswith('61 tool result c59: output 59')


def test_full_conversation_view_rejects_bad_options() -> None:
    with pytest.raises(ValueError, match='tools must be'):
        full_conversation_view(CODING_TRACE, budget=10_000, tools='verbose')  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match='tool_result cap'):
        full_conversation_view(CODING_TRACE, budget=10_000, tool_max_chars=-2)
    with pytest.raises(ValueError, match='user cap'):
        full_conversation_view(CODING_TRACE, budget=10_000, user_chars=0)
    with pytest.raises(ValueError, match='unknown message kinds'):
        full_conversation_view(CODING_TRACE, budget=10_000, include={'tool'})  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match='unknown message kinds'):
        full_conversation_view(CODING_TRACE, budget=10_000, exclude={'thinking'})  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match='not a string'):
        full_conversation_view(CODING_TRACE, budget=10_000, exclude='user')  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match='reasoning cap'):
        full_conversation_view(CODING_TRACE, budget=10_000, reasoning_chars=0)


def test_tool_activity_view_shows_inputs_statuses_and_safe_diagnostics() -> None:
    [view] = tool_activity_chunks(CODING_TRACE)

    assert 'CALL Bash [error]: cd repo && uv run pytest tests/test_parser.py -q' in view
    assert '→ diagnostic: provider_error' in view
    assert 'CALL Bash [completed]: git push --force origin main' in view
    assert 'output: Error: 1 failed, 3 passed' in view  # shell output keeps an excerpt
    assert 'output: forced update' in view
    assert 'Looking at the parser' not in view  # assistant prose is left out


def test_tool_activity_omits_raw_result_bodies_but_keeps_diagnostic_category() -> None:
    secret = 'sk-live-super-secret-token'
    trace = _trace(
        {'role': 'user', 'content': 'Run the lookup.'},
        {'role': 'assistant', 'tool_calls': [_call('c1', 'Lookup', {'query': 'record'})]},
        {
            'role': 'tool',
            'tool_call_id': 'c1',
            'content': f'Error: permission denied; credential={secret}',
        },
    )

    [view] = tool_activity_chunks(trace)

    assert 'CALL Lookup [error]' in view
    assert 'diagnostic: permission_denied' in view
    assert secret not in view
    assert 'credential=' not in view


def test_views_of_a_trace_without_tools() -> None:
    trace = _trace({'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'})

    assert tool_activity_chunks(trace) == ['No tool calls.']
    assert tool_inventory(trace) == 'No tool calls.'


def test_tool_inventory_lists_tools_commands_and_skills() -> None:
    inventory = tool_inventory(CODING_TRACE)

    assert '- Bash x2' in inventory
    assert '- pytest x1' in inventory
    assert '- systematic-debugging x1' in inventory


def _busy_trace(calls: int) -> TraceRecord:
    messages: list[dict[str, Any]] = [{'role': 'user', 'content': 'THE ORIGINAL TASK'}]
    for index in range(calls):
        messages.append({
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call(f'c{index}', 'Read', {'file_path': f'f{index}.py'})],
        })
        messages.append({'role': 'tool', 'tool_call_id': f'c{index}', 'content': 'x' * 200})
    return _trace(*messages)


def test_tool_activity_preserves_all_calls_when_middle_cut_would_omit_records() -> None:
    trace = _busy_trace(40)
    full = len(tool_activity_chunks(trace, budget=10**9)[0])
    budget = int(full * 0.75)

    chunks = tool_activity_chunks(trace, budget=budget)

    assert sum(chunk.count('CALL Read [completed]:') for chunk in chunks) == 40
    assert all(any(f'f{index}.py' in chunk for chunk in chunks) for index in range(40))
    assert all(len(chunk) <= budget for chunk in chunks)


def test_tool_activity_splits_a_long_trace_into_chunks_that_each_keep_the_task() -> None:
    trace = _busy_trace(40)
    full = len(tool_activity_chunks(trace, budget=10**9)[0])

    chunks = tool_activity_chunks(trace, budget=full // 3)

    assert len(chunks) >= 3
    assert all(chunk.startswith('USER: THE ORIGINAL TASK') and len(chunk) <= full // 3 for chunk in chunks)
    assert f'[part 1 of {len(chunks)} ' in chunks[0]
    assert sum(chunk.count('CALL Read') for chunk in chunks) == 40


def _failing_pytest_trace(output: str) -> TraceRecord:
    return _trace(
        {'role': 'user', 'content': 'Fix the parser.'},
        {'role': 'assistant', 'tool_calls': [_call('c1', 'Bash', {'command': 'uv run pytest -q'})]},
        {'role': 'tool', 'tool_call_id': 'c1', 'status': 'success', 'content': output},
        {'role': 'assistant', 'tool_calls': [_call('c2', 'Bash', {'command': 'uv run pytest -q'})]},
        {'role': 'tool', 'tool_call_id': 'c2', 'status': 'success', 'content': '7 passed, 0 failed'},
    )


def test_tool_activity_labels_a_failure_inside_a_completed_shell_call_with_a_scrubbed_excerpt() -> None:
    secret = 'sk-live-super-secret-token'
    trace = _failing_pytest_trace(
        f'FAILED tests/test_parser.py::test_empty\n{"." * 400}\nTOKEN={secret}\n{"." * 400}\n2 failed, 5 passed'
    )
    [view] = tool_activity_chunks(trace)

    assert 'output shows: tests_failed' in view
    assert view.count('output shows') == 1  # '0 failed' is not a failure
    assert 'output: FAILED tests/test_parser.py::test_empty' in view  # start of the output
    assert '2 failed, 5 passed' in view  # end of the output
    assert secret not in view  # the middle is cut
    assert 'output: 7 passed, 0 failed' in view


@pytest.mark.parametrize('cut', ['start', 'end'])
@pytest.mark.parametrize('padding', [100, 12_000], ids=['whole', 'two-windows'])
def test_a_secret_across_the_excerpt_cut_leaves_no_fragment_in_the_view(cut: str, padding: int) -> None:
    """The output is scrubbed in full before it is cut, so no head or tail of a credential survives the cut."""
    secret = 'ghp_' + 'aB3dE5gH7j' * 3
    middle = 'm' * (2 * padding)
    if cut == 'start':  # characters 130-170, across the 150th
        output = 'a' * 129 + ' ' + secret + ' ' + middle + ' end'
    else:  # ends 121 characters before the end, across the 150th from the end
        output = 'start ' + middle + ' ' + secret + ' ' + 'z' * 120
    trace = _failing_pytest_trace(output)

    [view] = tool_activity_chunks(trace)

    assert 'output: ' in view
    for size in (4, 8, 12):
        assert secret[:size] not in view
        assert secret[-size:] not in view


@pytest.mark.parametrize(
    ('tool', 'content'),
    [
        ('Bash', 'Exit code 1\nnpm ERR! missing script'),  # Claude Code
        ('bash', 'Exit code 1\nnpm ERR! missing script'),  # pi, omp
        ('exec_command', '{"chunk_id":"a1","exit_code":2,"output":"boom"}'),  # Codex
        ('orq_shell', '{"stdout":"","stderr":"boom","outcome":{"type":"exit","exit_code":2}}'),  # orq agents
    ],
)
def test_nonzero_exit_is_recognised_in_each_harness_format(tool: str, content: str) -> None:
    trace = _trace(
        {'role': 'user', 'content': 'Build it.'},
        {'role': 'assistant', 'tool_calls': [_call('c1', tool, {'command': 'make'})]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': content},
    )

    [view] = tool_activity_chunks(trace)

    assert 'output shows: nonzero_exit' in view


def test_codex_exec_script_is_read_as_a_shell_call() -> None:
    script = 'const r = await tools.exec_command({cmd:"git push --force origin main","workdir":"/repo"}); text(r);'
    trace = _trace(
        {'role': 'user', 'content': 'Ship it.'},
        {'role': 'assistant', 'tool_calls': [_call('c1', 'exec', script)]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Script failed\nOutput:\nrejected'},
    )

    [view] = tool_activity_chunks(trace)

    assert 'CALL exec [' in view
    assert ': git push --force origin main' in view
    assert 'output shows: nonzero_exit' in view
    assert 'output: Script failed' in view


def test_failure_markers_only_apply_to_shell_calls() -> None:
    trace = _trace(
        {'role': 'user', 'content': 'Read the log.'},
        {'role': 'assistant', 'tool_calls': [_call('c1', 'Read', {'path': 'log.txt'})]},
        {'role': 'tool', 'tool_call_id': 'c1', 'status': 'success', 'content': 'Traceback (most recent call last):'},
    )

    [view] = tool_activity_chunks(trace)

    assert 'output shows' not in view
    assert 'result body omitted' in view


@pytest.mark.parametrize(
    ('result', 'expected'),
    [({'records': [{'name': 'alpha'}]}, '"name": "alpha"'), ([{'id': 7}, 'beta'], '"id": 7')],
)
def test_full_conversation_renders_structured_tool_results(result: Any, expected: str) -> None:
    trace = _trace(
        {'role': 'user', 'content': 'Find the record.'},
        {'role': 'assistant', 'tool_calls': [_call('lookup-1', 'lookup', {'query': 'record'})]},
        {'role': 'tool', 'tool_call_id': 'lookup-1', 'content': result},
    )

    rendered = full_conversation_view(trace, tools='per_call', tool_max_chars=-1)

    assert 'tool result lookup-1:' in rendered
    assert expected in rendered

def test_full_conversation_renders_structured_and_multimodal_message_content() -> None:
    structured = {'text': 'label', 'data': {'records': [{'id': 7}]}, 'url': '/records/7'}
    trace = _trace(
        {'role': 'user', 'content': structured},
        {
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': 'typed text'},
                {'type': 'input_image', 'image_url': 'https://example.test/image.png'},
            ],
        },
    )

    rendered = full_conversation_view(trace)

    assert '"label"' in rendered
    assert '"records"' in rendered
    assert '"/records/7"' in rendered
    assert 'typed text' in rendered
    assert '[image]' in rendered


def test_full_conversation_source_spans_keep_message_indexes() -> None:
    literal_marker = '[... 100000 chars left out ...]'
    trace = _trace(
        {'role': 'user', 'content': literal_marker},
        {'role': 'assistant', 'content': '   '},
        {'role': 'assistant', 'content': 'final answer'},
    )

    rendered, spans, source_message_count = full_conversation_view_with_message_spans(trace)

    assert source_message_count == 3
    assert [index for _, _, index in spans] == [0, 2]
    assert literal_marker in rendered[spans[0][0] : spans[0][1]]
    assert 'final answer' in rendered[spans[1][0] : spans[1][1]]


def _header(text: str) -> str:
    first = text.splitlines()[0]
    assert first.startswith('[context: ') and first.endswith(']')
    return first


def test_coverage_header_is_off_by_default_and_changes_nothing() -> None:
    plain = full_conversation_view(CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=50)

    assert full_conversation_view(CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=50, coverage_header=False) == plain
    with_header = full_conversation_view(
        CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=50, coverage_header=True
    )
    assert with_header.startswith('[context: ')
    assert with_header.endswith(plain)
    assert full_conversation_view(CODING_TRACE, budget=10_000) == full_conversation_view(
        CODING_TRACE, budget=10_000, coverage_header=False
    )


def test_coverage_header_comes_first_and_lists_left_out_kinds_caps_and_message_count() -> None:
    view = full_conversation_view(
        CODING_TRACE,
        budget=10_000,
        tools='per_call',
        tool_max_chars=50,
        user_chars=200,
        assistant_edge=30,
        tool_input_chars=80,
        exclude=('system',),
        coverage_header=True,
    )

    header = _header(view)
    assert view.splitlines()[1].startswith('0 user:')
    assert 'includes user, assistant, tool input, tool result' in header
    assert 'left out: system, reasoning' in header
    assert 'user 200 chars' in header
    assert 'assistant 30 chars at each end' in header
    assert 'tool input 80 chars' in header
    assert 'tool result 50 chars' in header
    assert 'tools: one line per call' in header
    assert '7 source messages' in header


def test_coverage_header_reports_whole_for_unlimited_caps() -> None:
    header = _header(
        full_conversation_view(
            CODING_TRACE, budget=10_000, user_chars=-1, assistant_edge=-1, coverage_header=True
        )
    )

    assert 'user whole' in header
    assert 'assistant whole' in header


def test_coverage_header_names_excluded_reasoning_and_tool_results() -> None:
    header = _header(
        full_conversation_view(
            CODING_TRACE, budget=10_000, tools='per_call', tool_max_chars=50, exclude=('tool_result',), coverage_header=True
        )
    )

    assert 'left out: reasoning, tool result' in header
    assert 'tool result 50' not in header


def test_coverage_header_says_tool_results_are_left_out_when_the_cap_is_zero_or_tools_collapse() -> None:
    zero = _header(full_conversation_view(CODING_TRACE, budget=10_000, tools='per_call', coverage_header=True))
    collapsed = _header(full_conversation_view(CODING_TRACE, budget=10_000, tools='collapsed', coverage_header=True))

    assert 'tool result' in zero.split('left out:')[1].split(';')[0]
    assert 'one line per call' in zero
    assert 'tool result' in collapsed.split('left out:')[1].split(';')[0]
    assert 'collapsed to one `tools:` line per run, results left out' in collapsed


def test_coverage_header_separates_kinds_the_trace_lacks_from_kinds_left_out() -> None:
    header = _header(full_conversation_view(CODING_TRACE, budget=10_000, coverage_header=True))

    assert 'none in this trace: system' in header
    assert 'system' not in header.split('left out:')[1].split(';')[0]


def test_coverage_header_counts_against_the_budget_and_is_never_the_part_cut() -> None:
    trace = _trace(*({'role': 'user', 'content': f'turn {n} ' + 'x' * 400} for n in range(10)))
    plain = full_conversation_view(trace, budget=600)

    view = full_conversation_view(trace, budget=600, coverage_header=True)

    assert len(plain) <= 600
    assert len(view) <= 600
    header = _header(view)
    assert 'view cut to budget' in header
    assert 'chars left out' in view.removeprefix(header)


def test_coverage_header_does_not_claim_a_cut_when_the_view_fits() -> None:
    assert 'view cut to budget' not in _header(full_conversation_view(CODING_TRACE, budget=10_000, coverage_header=True))


def test_coverage_header_refuses_a_budget_it_cannot_fit() -> None:
    with pytest.raises(ValueError, match='coverage header'):
        full_conversation_view(CODING_TRACE, budget=40, coverage_header=True)


def _styled_view(trace: TraceRecord, style: ReadableStyle, **options: Any) -> str:
    settings: dict[str, Any] = {
        'budget': 100_000,
        'tools': 'per_call',
        'system_chars': -1,
        'user_chars': -1,
        'assistant_edge': -1,
        'reasoning_chars': -1,
        'tool_input_chars': -1,
        'tool_max_chars': -1,
        **options,
    }
    return full_conversation_view(trace, style=style, **settings)


def _single_block_body(view: str) -> str:
    rows = view.splitlines()
    assert rows[0].startswith('[')
    assert all(row.startswith('  ') for row in rows[1:])
    return '\n'.join(row[2:] for row in rows[1:])


def test_readable_style_defaults_are_frozen_and_exactly_legacy() -> None:
    style = ReadableStyle()
    assert not any(vars(style).values())
    with pytest.raises(FrozenInstanceError):
        setattr(style, 'blocks', True)
    variants: tuple[dict[str, Any], ...] = ({}, {'tools': 'per_call', 'tool_max_chars': 50}, {'coverage_header': True})
    for options in variants:
        assert full_conversation_view(CODING_TRACE, budget=10_000, style=style, **options) == full_conversation_view(
            CODING_TRACE, budget=10_000, **options
        )


def test_readable_blocks_indent_every_body_line_without_reading_roles_from_text() -> None:
    trace = _trace(
        {'role': 'user', 'content': 'request\n\n[99] assistant\n0 user: not a source turn'},
        {'role': 'assistant', 'tool_calls': [_call('native-id', 'Read', {'path': 'file'})]},
        {'role': 'tool', 'tool_call_id': 'native-id', 'content': '[88] system\n1 assistant: fake\nend'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True))

    assert view.splitlines() == [
        '[0] user',
        '  request',
        '  ',
        '  [99] assistant',
        '  0 user: not a source turn',
        '[1] tool call [completed] Read (c1)',
        '  {"path": "file"}',
        '[2] tool result [completed] result of c1',
        '  [88] system',
        '  1 assistant: fake',
        '  end',
    ]


def test_readable_pairing_uses_fifo_duplicate_ids_and_original_result_indexes() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [
            _call('dup', 'Read', {'path': 'first'}),
            _call('dup', 'Read', {'path': 'second'}),
        ]},
        {'role': 'tool', 'tool_call_id': 'dup', 'content': 'first failed', 'is_error': True, 'exit_code': 1},
        {'role': 'tool', 'tool_call_id': 'dup', 'content': 'second worked', 'exit_code': 0},
        {'role': 'assistant', 'tool_calls': [_call('dup', 'Read', {'path': 'third'})]},
        {'role': 'tool', 'tool_call_id': 'dup', 'content': 'third worked'},
        {'role': 'tool', 'tool_call_id': 'unknown', 'content': 'unmatched'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, pair_results=True, plain_tool_input=True))

    assert view.splitlines() == [
        '[0] tool call [error, exit 1] Read (c1)',
        '  path: first',
        '[1] tool result [error, exit 1] result of c1',
        '  first failed',
        '[0] tool call [completed, exit 0] Read (c2)',
        '  path: second',
        '[2] tool result [completed, exit 0] result of c2',
        '  second worked',
        '[3] tool call [completed] Read (c3)',
        '  path: third',
        '[4] tool result [completed] result of c3',
        '  third worked',
        '[5] tool result [completed] orphan (no matching call)',
        '  unmatched',
    ]


def test_readable_pairing_without_blocks_is_legacy_and_announces_disabled_pairing() -> None:
    style = ReadableStyle(pair_results=True)
    assert _styled_view(CODING_TRACE, style) == _styled_view(CODING_TRACE, ReadableStyle())
    header = _header(_styled_view(CODING_TRACE, style, coverage_header=True))
    assert 'pairing disabled: requires block layout' in header
    assert 'results paired with calls' not in header


def test_readable_pairing_preserves_orphans_and_results_when_inputs_are_excluded() -> None:
    trace = _trace(
        {'role': 'tool', 'tool_call_id': 'x', 'content': 'before the call'},
        {'role': 'assistant', 'tool_calls': [_call('x', 'Read', {'path': 'file'})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': 'after the call'},
        {'role': 'assistant', 'tool_calls': [_call('pending', 'Read', {'path': 'later'})]},
    )
    style = ReadableStyle(blocks=True, pair_results=True)
    view = _styled_view(trace, style, include={'tool_result'})
    assert view.splitlines() == [
        '[0] tool result [completed] orphan (no matching call)',
        '  before the call',
        '[2] tool result [completed] result of c1',
        '  after the call',
    ]
    with_calls = _styled_view(trace, style)
    assert '[3] tool call [pending] Read (c2)' in with_calls


@pytest.mark.parametrize(
    ('result', 'outcome'),
    [
        ({'content': 'ok', 'status': 'success', 'exit_code': 0}, 'completed, exit 0'),
        ({'content': 'Error: denied'}, 'error'),
        ({'content': 'failure', 'is_error': True}, 'error'),
        ({'content': 'failure', 'trace_finder_metadata': {'tool_result': {
            'status': 'failed', 'exit_code': 3, 'error_code': 'permission_denied',
        }}}, 'error, exit 3, error permission_denied'),
        ({'content': 'Exit code 7\nfailed'}, 'error, exit 7'),
        ({'content': '{"exit_code":2,"output":"failed"}'}, 'error, exit 2'),
        ({'content': '{"outcome":{"type":"exit","exit_code":4}}'}, 'error, exit 4'),
    ],
)
def test_readable_blocks_put_outcomes_before_tool_names_and_before_body_cuts(
    result: dict[str, Any], outcome: str,
) -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', 'Bash', {'command': 'run'})]},
        {'role': 'tool', 'tool_call_id': 'x', **result},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True), tool_max_chars=1)
    assert f'[0] tool call [{outcome}] Bash (c1)' in view
    assert f'[1] tool result [{outcome}] result of c1' in view


def test_readable_call_header_does_not_hide_a_later_result_failure() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', 'Read', {})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': 'part one', 'exit_code': 0},
        {'role': 'tool', 'tool_call_id': 'x', 'content': 'part two', 'exit_code': 2},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, pair_results=True))
    assert '[0] tool call [error, exit 2; completed, exit 0] Read (c1)' in view
    assert '[1] tool result [completed, exit 0] result of c1' in view
    assert '[2] tool result [error, exit 2] result of c1' in view


@pytest.mark.parametrize(
    ('arguments', 'expected'),
    [
        ('{"path":"a","options":{"x":1,"y":[true,null]},"count":2}', 'path: a\noptions: {"x":1,"y":[true,null]}\ncount: 2'),
        ('{"command":"first\\nsecond","label":"café"}', 'command: first\nsecond\nlabel: café'),
        ('{}', ''),
        ('["not","an","object"]', '["not","an","object"]'),
        ('"literal"', '"literal"'),
        ('{"path": "cut', '{"path": "cut'),
        ('raw tool command', 'raw tool command'),
    ],
)
def test_readable_plain_inputs_render_objects_and_preserve_raw_fallback(arguments: str, expected: str) -> None:
    trace = _trace({'role': 'assistant', 'tool_calls': [_call('x', 'Read', arguments)]})
    view = _styled_view(trace, ReadableStyle(blocks=True, plain_tool_input=True))
    assert _single_block_body(view) == expected


@pytest.mark.parametrize('kind', ['system', 'user', 'assistant', 'reasoning', 'tool_input', 'tool_result'])
@pytest.mark.parametrize('limit', [80, 101])
def test_readable_labelled_cuts_fit_part_caps_and_report_exact_omissions(kind: str, limit: int) -> None:
    source = 'HEAD' + 'x' * 500 + 'TAIL'
    messages: dict[str, tuple[dict[str, Any], ...]] = {
        'system': ({'role': 'system', 'content': source},),
        'user': ({'role': 'user', 'content': source},),
        'assistant': ({'role': 'assistant', 'content': source},),
        'reasoning': ({'role': 'assistant', 'reasoning_content': source},),
        'tool_input': ({'role': 'assistant', 'tool_calls': [_call('x', 'Read', source)]},),
        'tool_result': ({'role': 'tool', 'tool_call_id': 'orphan', 'content': source},),
    }
    caps = {
        'system': 'system_chars', 'user': 'user_chars', 'assistant': 'assistant_edge',
        'reasoning': 'reasoning_chars', 'tool_input': 'tool_input_chars', 'tool_result': 'tool_max_chars',
    }
    options = {'include': {kind}, caps[kind]: limit // 2 if kind == 'assistant' else limit}
    actual_limit = 2 * (limit // 2) if kind == 'assistant' else limit
    view = _styled_view(_trace(*messages[kind]), ReadableStyle(blocks=True, labelled_cuts=True), **options)
    body = _single_block_body(view)
    marker = re.search(r'\[\.\.\. (\d+) chars of (.*?) left out \.\.\.\]', body)
    assert marker is not None
    assert marker[2] == kind.replace('_', ' ')
    assert len(body) <= actual_limit
    assert int(marker[1]) + len(body) - len(marker[0]) == len(source)
    assert body.startswith('HEAD')
    if kind == 'tool_input':
        assert body.endswith(marker[0])
    else:
        assert body.endswith('TAIL')


def test_readable_labelled_cuts_do_not_use_legacy_slack_and_keep_whole_options() -> None:
    source = 'u' * 105
    trace = _trace({'role': 'user', 'content': source})
    legacy = _styled_view(trace, ReadableStyle(), user_chars=100)
    assert legacy == '0 user: ' + source
    cut = _styled_view(trace, ReadableStyle(blocks=True, labelled_cuts=True), user_chars=100)
    assert len(_single_block_body(cut)) <= 100
    assert 'chars of user left out' in cut
    whole = _styled_view(trace, ReadableStyle(blocks=True, labelled_cuts=True), user_chars=-1)
    assert _single_block_body(whole) == source


@pytest.mark.parametrize('cap', [1, 20, 35])
def test_readable_labelled_cuts_reject_unrepresentable_markers_with_a_safe_error(cap: int) -> None:
    secret = 'ghp_' + 'aB3dE5gH7j' * 3
    trace = _trace({'role': 'user', 'content': 'x' * 500 + secret})
    with pytest.raises(ValueError, match='^the message cap is too small for a labelled omission marker$'):
        _styled_view(trace, ReadableStyle(labelled_cuts=True), user_chars=cap)


def test_readable_terminal_noise_reduction_is_optional_and_counts_repeated_lines() -> None:
    output = '\x1b[31mstart\x1b[0m\rprogress\rfinished\r\nrepeat\nrepeat\nrepeat\n\nend'
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', 'Bash', {'command': 'echo \x1b[32mhello\x1b[0m'})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': output},
    )
    legacy = _styled_view(trace, ReadableStyle())
    assert output in legacy
    view = _styled_view(trace, ReadableStyle(blocks=True, plain_tool_input=True, strip_terminal_noise=True))
    assert '  command: echo hello' in view
    assert '  finished\n  repeat\n  (repeated 3 times)\n  \n  end' in view
    assert '\x1b' not in view and '\r' not in view and 'progress' not in view and 'start' not in view


def test_readable_terminal_noise_does_not_fold_two_lines_or_blank_runs() -> None:
    trace = _trace({'role': 'tool', 'content': 'twice\ntwice\n\n\n\nlast'})
    body = _single_block_body(_styled_view(trace, ReadableStyle(blocks=True, strip_terminal_noise=True)))
    assert body == 'twice\ntwice\n\n\n\nlast'


def test_readable_cwd_shortening_requires_evidence_and_respects_path_boundaries() -> None:
    cwd = '/workspace/project'
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', 'Read', {'cwd': cwd, 'path': cwd + '/a.py'})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': f'{cwd}/a.py {cwd} {cwd}ile/b.py /elsewhere/c.py'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, plain_tool_input=True, shorten_cwd=True))
    assert f'paths under {cwd} are shown as ./' in view
    assert '  cwd: .\n  path: ./a.py' in view
    assert '  ./a.py . /workspace/projectile/b.py /elsewhere/c.py' in view
    assert cwd + '/a.py' in _styled_view(trace, ReadableStyle())


def test_readable_cwd_shortening_never_guesses_from_common_absolute_paths() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [
            _call('a', 'Read', {'path': '/workspace/project/a.py'}),
            _call('b', 'Read', {'path': '/workspace/project/b.py'}),
        ]},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, plain_tool_input=True, shorten_cwd=True))
    assert '/workspace/project/a.py' in view and '/workspace/project/b.py' in view
    assert 'paths under' not in view


@pytest.mark.parametrize('other', ['/different/repo', 'relative/directory', '/'])
def test_readable_cwd_shortening_disables_conflicting_or_unreliable_evidence(other: str) -> None:
    trace = _trace({'role': 'assistant', 'tool_calls': [
        _call('a', 'Read', {'cwd': '/workspace/project', 'path': '/workspace/project/a.py'}),
        _call('b', 'Read', {'workdir': other}),
    ]})
    view = _styled_view(trace, ReadableStyle(shorten_cwd=True))
    assert '/workspace/project/a.py' in view and 'paths under' not in view


@pytest.mark.parametrize('local', [False, True])
def test_readable_cwd_shortening_uses_only_explicit_capture_or_local_project_metadata(local: bool) -> None:
    cwd = '/workspace/project'
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', 'Read', {'path': cwd + '/a.py'})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': cwd + '/a.py'},
    )
    trace = trace.model_copy(update=(
        {'project': cwd, 'capture_metadata': {'source': 'local:omp'}}
        if local else {'capture_metadata': {'cwd': cwd}}
    ))
    view = _styled_view(trace, ReadableStyle(shorten_cwd=True))
    assert './a.py' in view
    ordinary_project = trace.model_copy(update={'capture_metadata': {}, 'project': cwd})
    assert 'paths under' not in _styled_view(ordinary_project, ReadableStyle(shorten_cwd=True))


@pytest.mark.parametrize('pair', [False, True])
def test_readable_repeated_calls_keep_every_distinct_result_and_reference_identical_results(pair: bool) -> None:
    output = 'a line of useful output\n' * 5
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('a', 'Read', {'path': 'file'})]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': output},
        {'role': 'assistant', 'tool_calls': [_call('b', 'Read', {'path': 'file'})]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': output},
        {'role': 'assistant', 'tool_calls': [_call('c', 'Read', {'path': 'file'})]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'a different result'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, pair_results=pair, collapse_repeats=True))
    assert view.count('tool call') == 1
    assert '[0] tool call [completed] Read (c1-c3) x3' in view
    assert '[1] tool result [completed] result of c1' in view
    assert '[3] tool result [completed] result of c2\n  same as result [1] (c1)' in view
    assert '[5] tool result [completed] result of c3\n  a different result' in view
    assert view.count('tool result') == 3
    assert _styled_view(trace, ReadableStyle()).count('tool input') == 3


def test_readable_call_runs_do_not_cross_excluded_source_prose_or_nonidentical_calls() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('a', 'Read', {'path': 'file'})]},
        {'role': 'assistant', 'content': 'new context'},
        {'role': 'assistant', 'tool_calls': [_call('b', 'Read', {'path': 'file'}), _call('c', 'Read', {'path': 'other'})]},
        {'role': 'assistant', 'tool_calls': [_call('d', 'Read', {'path': 'file'})]},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, collapse_repeats=True), include={'tool_input'})
    assert view.count('tool call') == 4
    assert ' x2' not in view


def test_readable_references_keep_original_indexes_even_for_duplicate_ids_and_orphans() -> None:
    output = 'a sufficiently long identical result to make a reference useful'
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('dup', 'Read', {'path': 'first'}), _call('dup', 'Read', {'path': 'second'})]},
        {'role': 'tool', 'tool_call_id': 'dup', 'content': output},
        {'role': 'tool', 'tool_call_id': 'dup', 'content': output, 'is_error': True},
        {'role': 'tool', 'tool_call_id': 'orphan', 'content': output},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, pair_results=True, collapse_repeats=True))
    assert '[2] tool result [error] result of c2\n  same as result [1] (c1)' in view
    assert '[3] tool result [completed] orphan (no matching call)\n  same as result [1] (c1)' in view
    assert view.count('tool call') == 2


def test_readable_blob_reduction_is_optional_and_never_merges_distinct_outputs() -> None:
    first, second = 'ABCDEFGH' * 40, 'HGFEDCBA' * 40
    digest = '0123456789abcdef' * 5
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('a', 'Read', {}), _call('b', 'Read', {})]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': f'{first}\nsha256:{digest}\n{"=" * 250}'},
        {'role': 'tool', 'tool_call_id': 'b', 'content': f'{second}\nsha256:{digest}\n{"=" * 250}'},
    )
    plain = _styled_view(trace, ReadableStyle())
    assert first in plain and second in plain and digest in plain
    view = _styled_view(trace, ReadableStyle(blocks=True, collapse_blobs=True, collapse_repeats=True))
    assert view.count('[blob, 320 chars]') == 2
    assert view.count('sha256:[blob, 80 chars]') == 2
    assert '=' * 250 in view
    assert 'same as result' not in view
    assert first not in view and second not in view


def test_readable_result_references_do_not_compare_only_truncated_edges() -> None:
    trace = _trace(
        {'role': 'tool', 'content': 'HEAD' + 'a' * 500 + 'TAIL'},
        {'role': 'tool', 'content': 'HEAD' + 'b' * 500 + 'TAIL'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, labelled_cuts=True, collapse_repeats=True), tool_max_chars=60)
    assert 'same as result' not in view
    assert view.count('chars of tool result left out') == 2


def test_readable_reductions_and_references_scrub_secrets_before_shortening() -> None:
    secret = 'ghp_' + 'aB3dE5gH7j' * 3
    split_secret = secret[:4] + '\x1b[31m' + secret[4:] + '\x1b[0m'
    output = f'first {split_secret}\n' + 'a useful result line\n' * 8
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [
            _call('a', 'Read', {'token': secret, 'path': '/workspace/project/file'}),
            _call('b', 'Read', {'token': secret, 'path': '/workspace/project/file'}),
        ]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': output, 'error_code': secret},
        {'role': 'tool', 'tool_call_id': 'b', 'content': output},
    )
    style = ReadableStyle(
        blocks=True, pair_results=True, plain_tool_input=True, labelled_cuts=True,
        strip_terminal_noise=True, shorten_cwd=True, collapse_repeats=True, collapse_blobs=True,
    )
    view = _styled_view(trace, style, tool_input_chars=80, tool_max_chars=80)
    assert '<API_KEY>' in view
    for fragment in (secret, secret[:8], secret[-8:]):
        assert fragment not in view
    assert 'same as result [1] (c1)' in view


def test_readable_header_names_enabled_transformations_and_total_budget_stays_bounded() -> None:
    trace = _trace(*({'role': 'user', 'content': 'A long paragraph with ordinary words and spaces. ' * 30} for _ in range(20)))
    style = ReadableStyle(
        blocks=True, pair_results=True, plain_tool_input=True, labelled_cuts=True,
        strip_terminal_noise=True, shorten_cwd=True, collapse_repeats=True, collapse_blobs=True,
    )
    view = _styled_view(trace, style, coverage_header=True, budget=1_400)
    header = _header(view)
    assert 'view cut to budget' in header
    assert len(view) <= 1_400
    assert 'chars left out' in view.removeprefix(header)


def test_readable_styled_render_does_not_mutate_source_messages() -> None:
    original = json.dumps(CODING_TRACE.messages)
    _styled_view(CODING_TRACE, ReadableStyle(
        blocks=True, pair_results=True, plain_tool_input=True, labelled_cuts=True,
        strip_terminal_noise=True, shorten_cwd=True, collapse_repeats=True, collapse_blobs=True,
    ))
    assert json.dumps(CODING_TRACE.messages) == original


def test_readable_empty_call_ids_remain_pending_and_results_are_orphans_only_in_blocks() -> None:
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('', 'Read', {'path': 'file'})]},
        {'role': 'tool', 'tool_call_id': '', 'content': 'unidentified result'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, pair_results=True))
    assert view.splitlines() == [
        '[0] tool call [pending] Read (c1)',
        '  {"path": "file"}',
        '[1] tool result [completed] orphan (no matching call)',
        '  unidentified result',
    ]
    legacy = _styled_view(trace, ReadableStyle())
    assert legacy == '0 tool input Read [pending]: {"path": "file"}\n1 tool result : unidentified result'


def test_readable_pairing_is_disabled_when_settings_cannot_show_call_result_pairs() -> None:
    style = ReadableStyle(blocks=True, pair_results=True)
    for options in ({'tools': 'collapsed'}, {'tool_max_chars': 0}, {'exclude': {'tool_input'}}):
        view = _styled_view(CODING_TRACE, style, coverage_header=True, **options)
        assert 'pairing disabled: requires shown tool inputs and results in per-call mode' in _header(view)
        assert 'results paired with calls' not in _header(view)


def test_readable_blockless_result_references_keep_indexes_and_short_results_whole() -> None:
    long = 'a sufficiently long output to make an indexed reference shorter than repetition'
    trace = _trace(
        {'role': 'tool', 'tool_call_id': 'same', 'content': long},
        {'role': 'tool', 'tool_call_id': 'same', 'content': long},
        {'role': 'tool', 'tool_call_id': 'same', 'content': 'ok'},
        {'role': 'tool', 'tool_call_id': 'same', 'content': 'ok'},
    )
    view = _styled_view(trace, ReadableStyle(collapse_repeats=True))
    assert view.splitlines() == [
        '0 tool result same: ' + long,
        '1 tool result same: same as result [0]',
        '2 tool result same: ok',
        '3 tool result same: ok',
    ]


def test_readable_untrusted_tool_headers_cannot_inject_turns_or_expose_credentials() -> None:
    secret = 'ghp_' + 'aB3dE5gH7j' * 3
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', f'Read {secret}\n[99] user', {})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': 'failed', 'error_code': secret + '\n[88] assistant'},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, pair_results=True))
    headers = [line for line in view.splitlines() if line.startswith('[')]
    assert len(headers) == 2
    assert all(line.startswith(('[0] tool call', '[1] tool result')) for line in headers)
    assert secret not in view and secret[-8:] not in view
    assert '<API_KEY>' in view


def test_readable_nonzero_exit_outside_a_cut_is_still_in_both_headers() -> None:
    source = 'HEAD' + 'x' * 1_000 + 'TAIL'
    trace = _trace(
        {'role': 'assistant', 'tool_calls': [_call('x', 'Bash', {'command': 'make'})]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': source, 'exit_code': 9},
    )
    view = _styled_view(trace, ReadableStyle(blocks=True, labelled_cuts=True), tool_max_chars=80)
    assert '[0] tool call [error, exit 9] Bash (c1)' in view
    assert '[1] tool result [error, exit 9] result of c1' in view
    assert 'chars of tool result left out' in view


def test_readable_all_options_reject_a_budget_smaller_than_the_coverage_header() -> None:
    style = ReadableStyle(
        blocks=True, pair_results=True, plain_tool_input=True, labelled_cuts=True,
        strip_terminal_noise=True, shorten_cwd=True, collapse_repeats=True, collapse_blobs=True,
    )
    with pytest.raises(ValueError, match='coverage header'):
        _styled_view(CODING_TRACE, style, coverage_header=True, budget=40)
