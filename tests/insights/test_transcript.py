"""Unit tests for `evaluatorq.insights.transcript` — the classifier's text views and tool counts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.insights.labeling import redacted_shell_outputs
from evaluatorq.insights.transcript import (
    conversation_view,
    first_command,
    is_risky_command,
    tool_activity_chunks,
    tool_inventory,
    tool_stats,
)
from evaluatorq.trace_finder.models import TraceRecord
from tests.common.fake_orq import FakeOrq


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


def test_conversation_view_drops_outputs_and_reminders_and_shows_risky_commands_in_full() -> None:
    view = conversation_view(CODING_TRACE)

    assert view.startswith('USER: Fix the failing parser test.')
    assert 'injected context' not in view
    assert '1 failed' not in view  # tool output
    assert 'Bash: pytest' in view
    assert 'Bash: `git push --force origin main`' in view
    assert 'Skill: systematic-debugging' in view
    assert 'Edit: src/parser.py' in view
    assert view.endswith('ASSISTANT: Fixed and pushed.')


def test_conversation_view_keeps_the_opening_request_and_cuts_the_middle() -> None:
    turns: list[dict[str, Any]] = [{'role': 'user', 'content': 'THE ORIGINAL TASK'}]
    for index in range(200):
        turns.append({'role': 'assistant', 'content': f'step {index} ' + 'x' * 200})
        turns.append({'role': 'user', 'content': f'reply {index}'})
    view = conversation_view(_trace(*turns), budget=5_000)

    assert len(view) <= 5_000
    assert view.startswith('USER: THE ORIGINAL TASK')
    assert 'lines omitted' in view
    assert view.rstrip().endswith('reply 199')


def test_conversation_view_keeps_a_long_opening_request_over_budget() -> None:
    request = 'please ' * 500
    view = conversation_view(
        _trace({'role': 'user', 'content': request}, {'role': 'assistant', 'content': 'ok'}), budget=1_000
    )

    assert view.startswith('USER: please please')


@pytest.mark.asyncio
async def test_tool_activity_view_shows_inputs_statuses_and_safe_diagnostics() -> None:
    redacted = await redacted_shell_outputs(CODING_TRACE, orq=FakeOrq().client)

    [view] = tool_activity_chunks(CODING_TRACE, redacted_outputs=redacted)

    assert 'CALL Bash [error]: cd repo && uv run pytest tests/test_parser.py -q' in view
    assert '→ diagnostic: provider_error' in view
    assert 'CALL Bash [completed]: git push --force origin main' in view
    assert 'output: Error: 1 failed, 3 passed' in view  # shell output keeps an excerpt
    assert 'output: forced update' in view
    assert 'Looking at the parser' not in view  # assistant prose is left out


def test_tool_activity_view_without_redacted_outputs_renders_no_output_text() -> None:
    views = [
        *tool_activity_chunks(CODING_TRACE),
        *tool_activity_chunks(CODING_TRACE, redacted_outputs={'c2': None, 'c4': None}),
    ]

    for view in views:
        assert 'output:' not in view
        assert 'Error: 1 failed' not in view
        assert 'forced update' not in view
        assert 'diagnostic: provider_error' in view  # fixed labels are computed locally and stay


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


def test_tool_activity_cuts_the_middle_when_at_most_30_percent_is_lost() -> None:
    trace = _busy_trace(40)
    full = len(tool_activity_chunks(trace, budget=10**9)[0])

    [view] = tool_activity_chunks(trace, budget=int(full * 0.75))

    assert 'lines omitted' in view


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


@pytest.mark.asyncio
async def test_tool_activity_labels_a_failure_inside_a_completed_shell_call_with_a_redacted_excerpt() -> None:
    secret = 'sk-live-super-secret-token'
    trace = _failing_pytest_trace(
        f'FAILED tests/test_parser.py::test_empty\n{"." * 400}\nTOKEN={secret}\n{"." * 400}\n2 failed, 5 passed'
    )
    redacted = await redacted_shell_outputs(trace, orq=FakeOrq().client)

    [view] = tool_activity_chunks(trace, redacted_outputs=redacted)

    assert 'output shows: tests_failed' in view
    assert view.count('output shows') == 1  # '0 failed' is not a failure
    assert 'output: FAILED tests/test_parser.py::test_empty' in view  # start of the output
    assert '2 failed, 5 passed' in view  # end of the output
    assert secret not in view  # the middle is cut
    assert 'output: 7 passed, 0 failed' in view


@pytest.mark.asyncio
@pytest.mark.parametrize('cut', ['start', 'end'])
@pytest.mark.parametrize('padding', [100, 12_000], ids=['whole', 'two-windows'])
async def test_a_secret_across_the_excerpt_cut_leaves_no_fragment_in_the_view(cut: str, padding: int) -> None:
    """The output is redacted in full before it is cut, so no head or tail of a credential survives the cut."""
    secret = 'sk-live-' + 'ABCDEFGH' * 4
    middle = 'm' * (2 * padding)
    if cut == 'start':  # characters 130-170, across the 150th
        output = 'a' * 130 + secret + ' ' + middle + ' end'
    else:  # ends 121 characters before the end, across the 150th from the end
        output = 'start ' + middle + ' ' + secret + ' ' + 'z' * 120
    trace = _failing_pytest_trace(output)

    redacted = await redacted_shell_outputs(trace, orq=FakeOrq().client)
    [view] = tool_activity_chunks(trace, redacted_outputs=redacted)

    assert 'output: ' in view
    for size in (4, 8, 12):
        assert secret[:size] not in view
        assert secret[-size:] not in view


@pytest.mark.asyncio
async def test_a_long_output_is_redacted_from_its_two_ends_only() -> None:
    trace = _failing_pytest_trace('s' * 100 + ' ' + 'm' * 200_000 + ' ' + 'e' * 100)
    orq = FakeOrq()

    await redacted_shell_outputs(trace, orq=orq.client)

    sent = sum(len(request) for request in orq.requests)
    assert sent < 20_000


@pytest.mark.asyncio
async def test_unredactable_shell_output_renders_no_excerpt_but_keeps_the_failure_labels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = 'sk-live-super-secret-token'
    trace = _failing_pytest_trace(f'Exit code 1\nTOKEN={secret}\n3 failed')

    for orq in (FakeOrq(error=RuntimeError('down')).client, None):
        redacted = await redacted_shell_outputs(trace, orq=orq)
        [view] = tool_activity_chunks(trace, redacted_outputs=redacted)

        assert redacted['c1'] is None
        assert 'output shows: nonzero_exit, tests_failed' in view
        assert 'output: Exit code' not in view
        assert secret not in view
        assert 'Exit code 1' not in view
    assert 'PII redaction' in caplog.text


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


@pytest.mark.asyncio
async def test_codex_exec_script_is_read_as_a_shell_call() -> None:
    script = 'const r = await tools.exec_command({cmd:"git push --force origin main","workdir":"/repo"}); text(r);'
    trace = _trace(
        {'role': 'user', 'content': 'Ship it.'},
        {'role': 'assistant', 'tool_calls': [_call('c1', 'exec', script)]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Script failed\nOutput:\nrejected'},
    )

    redacted = await redacted_shell_outputs(trace, orq=FakeOrq().client)

    [view] = tool_activity_chunks(trace, redacted_outputs=redacted)

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
