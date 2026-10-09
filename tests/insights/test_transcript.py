"""Unit tests for `evaluatorq.insights.transcript` — the classifier's text views and tool counts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest
from evaluatorq.common.model_input import serialized_question_chars

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

    rendered = full_conversation_view(trace)

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
