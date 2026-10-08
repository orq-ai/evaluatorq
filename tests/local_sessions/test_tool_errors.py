"""A failed tool call in a local session reaches the Insights transcript as `[error]`, not `[completed]`."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evaluatorq.insights.transcript import tool_activity_chunks
from evaluatorq.local_sessions import session_document
from evaluatorq.local_sessions.claude import READER as CLAUDE
from evaluatorq.local_sessions.codex import READER as CODEX
from evaluatorq.local_sessions.items import function_call_output
from evaluatorq.local_sessions.omp import READER as OMP

from .conftest import parse_file, write_jsonl
from .test_claude import _assistant, _tool_result
from .test_claude import _user as claude_user
from .test_codex import _item as codex_item
from .test_codex import _message as codex_message
from .test_codex import _meta as codex_meta
from .test_codex import _rollout as codex_rollout
from .test_omp import _header as omp_header
from .test_omp import _msg as omp_msg
from .test_omp import _session as omp_session
from .test_omp import _user as omp_user

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.local_sessions.models import ParsedSession


def _activity(parsed: ParsedSession) -> str:
    return '\n'.join(tool_activity_chunks(session_document(parsed)))


def _assert_one_failed_one_fine(parsed: ParsedSession) -> None:
    text = _activity(parsed)
    failed, fine = (next(line for line in text.splitlines() if marker in line) for marker in ('FAILED', 'FINE'))
    assert failed.startswith('CALL Bash [error]:'), text
    assert fine.startswith('CALL Bash [completed]:'), text
    lines = text.splitlines()
    assert lines[lines.index(failed) + 1].startswith('  → diagnostic: '), text
    assert lines[lines.index(fine) + 1].startswith('  → output: '), text


def test_claude_is_error_renders_as_an_error(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 't.jsonl',
        [
            claude_user('u1', None, 0, 'run both'),
            _assistant(
                'a1',
                'u1',
                1,
                [{'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'command': 'FAILED'}}],
                mid='m1',
            ),
            claude_user(
                'r1', 'a1', 2, [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'boom', 'is_error': True}]
            ),
            _assistant(
                'a2',
                'r1',
                3,
                [{'type': 'tool_use', 'id': 't2', 'name': 'Bash', 'input': {'command': 'FINE'}}],
                mid='m2',
            ),
            _tool_result('r2', 'a2', 4, 't2', 'ok'),
        ],
    )

    _assert_one_failed_one_fine(parse_file(CLAUDE, path))


def test_omp_is_error_renders_as_an_error(omp_sessions: Path) -> None:
    call = {'type': 'toolCall', 'name': 'Bash'}
    path = omp_session(
        omp_sessions,
        [
            *omp_header(),
            omp_user('a', None, 1, 'run both'),
            omp_msg('b', 'a', 2, role='assistant', content=[{**call, 'id': 't1', 'arguments': {'command': 'FAILED'}}]),
            omp_msg(
                'c',
                'b',
                3,
                role='toolResult',
                toolCallId='t1',
                isError=True,
                content=[{'type': 'text', 'text': 'boom'}],
            ),
            omp_msg('d', 'c', 4, role='assistant', content=[{**call, 'id': 't2', 'arguments': {'command': 'FINE'}}]),
            omp_msg(
                'e',
                'd',
                5,
                role='toolResult',
                toolCallId='t2',
                isError=False,
                content=[{'type': 'text', 'text': 'ok'}],
            ),
        ],
    )

    _assert_one_failed_one_fine(parse_file(OMP, path))


@pytest.mark.parametrize(
    'failed_output',
    ['Script failed\nTraceback (most recent call last)', 'Chunk ID: 1\nExit code: 2\nOutput:\nnope'],
)
def test_codex_failed_calls_render_as_errors(codex_home: Path, failed_output: str) -> None:
    def call(minute: int, call_id: str, command: str) -> dict[str, object]:
        return codex_item(
            minute, type='function_call', call_id=call_id, name='Bash', arguments=f'{{"command": "{command}"}}'
        )

    path = codex_rollout(
        codex_home,
        records=[
            codex_meta(),
            codex_message(1, 'user', 'run both'),
            call(2, 'c1', 'FAILED'),
            codex_item(3, type='function_call_output', call_id='c1', output=failed_output),
            call(4, 'c2', 'FINE'),
            codex_item(5, type='function_call_output', call_id='c2', output='Chunk ID: 2\nExit code: 0\nOutput:\nok'),
        ],
    )

    _assert_one_failed_one_fine(parse_file(CODEX, path))


@pytest.mark.parametrize(
    'output',
    [
        'Process exited with code 1',
        '{"exit_code": 1, "output": "x"}',
        'Exit code: 0\nScript failed',
        'The Script failed to start but this is prose',
    ],
)
def test_codex_text_that_is_not_a_failure_marker_stays_completed(codex_home: Path, output: str) -> None:
    path = codex_rollout(
        codex_home,
        records=[
            codex_meta(),
            codex_message(1, 'user', 'run'),
            codex_item(2, type='function_call', call_id='c1', name='Bash', arguments='{"command": "x"}'),
            codex_item(3, type='function_call_output', call_id='c1', output=output),
        ],
    )

    assert 'CALL Bash [completed]' in _activity(parse_file(CODEX, path))


def test_error_prefix_survives_trimming() -> None:
    item = function_call_output(call_id='c', output='x' * 50_000, is_error=True)

    assert item['output'].startswith('Error: xxx')
    assert '\n[truncated 30007 chars]\n' in item['output']
    assert item['output'].endswith('xxx')
