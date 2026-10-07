"""ATIF documents for parsed local sessions."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.local_sessions import session_document
from evaluatorq.local_sessions.claude import READER

from .conftest import write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

_BASE = {'sessionId': 'sess-9', 'cwd': '/w', 'version': '2.0.0', 'entrypoint': 'cli'}


def test_session_document_round_trips_through_atif(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-w' / 'sess-9.jsonl',
        [
            {
                **_BASE,
                'type': 'user',
                'uuid': 'u1',
                'parentUuid': None,
                'timestamp': '2026-10-01T10:00:00Z',
                'message': {'role': 'user', 'content': 'run ls'},
            },
            {
                **_BASE,
                'type': 'assistant',
                'uuid': 'a1',
                'parentUuid': 'u1',
                'timestamp': '2026-10-01T10:00:05Z',
                'message': {
                    'id': 'm1',
                    'model': 'claude-x',
                    'content': [
                        {'type': 'thinking', 'thinking': 'plan'},
                        {'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'c': 'ls'}},
                    ],
                    'usage': {'input_tokens': 4, 'output_tokens': 6},
                },
            },
            {
                **_BASE,
                'type': 'user',
                'uuid': 'r1',
                'parentUuid': 'a1',
                'timestamp': '2026-10-01T10:00:06Z',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'a.txt'}],
                },
            },
            {
                **_BASE,
                'type': 'assistant',
                'uuid': 'a2',
                'parentUuid': 'r1',
                'timestamp': '2026-10-01T10:00:10Z',
                'message': {'id': 'm2', 'model': 'claude-x', 'content': [{'type': 'text', 'text': 'one file'}]},
            },
        ],
    )
    document = session_document(READER.parse(path))
    steps = document.trajectory.steps
    assert [step.source for step in steps] == ['user', 'agent', 'agent']
    first = steps[1]
    assert first.reasoning_content == 'plan'
    assert first.tool_calls is not None
    assert len(first.tool_calls) == 1
    assert first.observation is not None
    assert [result.source_call_id for result in first.observation.results] == ['t1']
    metadata = document.metadata
    assert metadata.trace_id == 'claude-code:sess-9'
    assert metadata.capture_metadata['source'] == 'local:claude-code'
    assert metadata.tool_names == ('Bash',)
    assert metadata.total_tokens == 10
    assert metadata.duration_ms == 10_000
    assert any(message.get('role') == 'tool' and message.get('tool_call_id') == 't1' for message in document.messages)
