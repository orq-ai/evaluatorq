"""omp reader."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.local_sessions import session_document
from evaluatorq.local_sessions.omp import READER

from .conftest import parse_file, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

SID = '0199aaaa-0000-7000-8000-000000000001'
NAME = f'2026-10-01T10-00-00-000Z_{SID}.jsonl'


def _ts(minute: int) -> str:
    return f'2026-10-01T10:{minute:02d}:00.000Z'


def _header(**extra: Any) -> list[dict[str, Any]]:
    return [
        {'type': 'title', 'title': 'Fix the bug', 'source': 'auto', 'v': 1},
        {'type': 'session', 'id': SID, 'cwd': '/work/proj', 'timestamp': _ts(0), 'version': '3', **extra},
    ]


def _entry(record_id: str, parent: str | None, minute: int, **fields: Any) -> dict[str, Any]:
    return {'id': record_id, 'parentId': parent, 'timestamp': _ts(minute), **fields}


def _msg(record_id: str, parent: str | None, minute: int, **message: Any) -> dict[str, Any]:
    return _entry(record_id, parent, minute, type='message', message=message)


def _user(record_id: str, parent: str | None, minute: int, text: str) -> dict[str, Any]:
    return _msg(record_id, parent, minute, role='user', content=[{'type': 'text', 'text': text}])


def _session(root: Path, records: list[dict[str, Any]], *, name: str = NAME) -> Path:
    return write_jsonl(root / '-proj' / name, records)


def _texts(items: list[dict[str, Any]]) -> list[str]:
    return [item['content'][0]['text'] for item in items if item['type'] == 'message']


def test_summary_fields(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [*_header(), _user('a', None, 1, 'fix the bug'), _msg('b', 'a', 5, role='assistant', content=[])],
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.source == 'omp'
    assert summary.session_id == SID
    assert summary.title == 'Fix the bug'
    assert summary.project_dir == '/work/proj'
    assert summary.agent_version == '3'
    assert summary.first_prompt == 'fix the bug'
    assert summary.started_at.isoformat() == '2026-10-01T10:00:00+00:00'
    assert summary.updated_at.isoformat() == '2026-10-01T10:05:00+00:00'


def test_session_title_is_the_fallback(omp_sessions: Path) -> None:
    records = [_header(title='From header')[1], _user('a', None, 1, 'hi')]
    summary = READER.summarize(_session(omp_sessions, records))
    assert summary is not None
    assert summary.title == 'From header'


def test_depth_three_subagent_is_not_a_session_path(omp_sessions: Path) -> None:
    top = _session(omp_sessions, [*_header(), _user('a', None, 1, 'hi')])
    nested = omp_sessions / '-proj' / 'sub' / NAME
    assert READER.is_session_path(top, omp_sessions)
    assert not READER.is_session_path(nested, omp_sessions)
    assert list(READER.iter_files(omp_sessions)) == [top]


def test_parent_session_is_not_a_main_session(omp_sessions: Path) -> None:
    path = _session(omp_sessions, [*_header(parentSession='/x.jsonl'), _user('a', None, 1, 'hi')])
    assert READER.summarize(path) is None


def test_session_without_a_prompt_is_listed_with_an_empty_prompt(omp_sessions: Path) -> None:
    summary = READER.summarize(_session(omp_sessions, _header()))
    assert summary is not None
    assert summary.first_prompt == ''


def test_session_whose_prompt_is_beyond_the_head_is_listed(omp_sessions: Path) -> None:
    path = _session(omp_sessions, [*_header(), _user('a', None, 1, 'x' * 300_000), _user('b', 'a', 2, 'later')])
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.first_prompt == ''
    assert summary.session_id == SID


def test_oversized_first_line_does_not_raise(omp_sessions: Path) -> None:
    path = _session(omp_sessions, [{'type': 'session', 'id': SID, 'pad': 'x' * 300_000}])
    assert READER.summarize(path) is None


def test_rewound_branch_keeps_only_the_last_leaf_chain(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'first'),
            _user('b', 'a', 2, 'abandoned'),
            _user('c', 'a', 3, 'rewound'),
        ],
    )
    assert _texts(parse_file(READER, path).items) == ['first', 'rewound']


def test_tool_call_and_result_become_a_tool_call_with_observation(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'list'),
            _msg(
                'b',
                'a',
                2,
                role='assistant',
                provider='anthropic',
                model='claude-x',
                usage={'totalTokens': 10},
                content=[
                    {'type': 'thinking', 'thinking': 'plan', 'thinkingSignature': 'sig'},
                    {'type': 'thinking', 'thinking': ''},
                    {
                        'type': 'toolCall',
                        'id': 't1',
                        'name': 'bash',
                        'arguments': {'cmd': 'ls'},
                        'intent': 'listing',
                        'thoughtSignature': 'sig2',
                    },
                ],
            ),
            _msg(
                'c',
                'b',
                3,
                role='toolResult',
                toolCallId='t1',
                toolName='bash',
                isError=False,
                content=[{'type': 'text', 'text': 'a.py'}],
            ),
            _msg(
                'd',
                'c',
                4,
                role='assistant',
                model='claude-x',
                usage={'totalTokens': 5},
                content=[{'type': 'text', 'text': 'done'}],
            ),
        ],
    )
    parsed = parse_file(READER, path)
    assert parsed.model == 'claude-x'
    assert parsed.total_tokens == 15
    assert [item['type'] for item in parsed.items].count('reasoning') == 1
    assert 'sig' not in str(parsed.items)
    document = session_document(parsed)
    calls = [call for step in document.trajectory.steps for call in step.tool_calls or []]
    assert [call.function_name for call in calls] == ['bash']
    assert calls[0].arguments == {'cmd': 'ls'}
    observations = [
        result.content for step in document.trajectory.steps if step.observation for result in step.observation.results
    ]
    assert observations == ['a.py']


def test_provider_prefixes_the_model(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'hi'),
            _msg(
                'b', 'a', 2, role='assistant', provider='openai', model='gpt-x', content=[{'type': 'text', 'text': 'x'}]
            ),
        ],
    )
    assert parse_file(READER, path).model == 'openai/gpt-x'


def test_errored_tool_result_is_prefixed_for_the_transcript(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'run'),
            _msg(
                'b',
                'a',
                2,
                role='assistant',
                content=[{'type': 'toolCall', 'id': 't1', 'name': 'bash', 'arguments': {}}],
            ),
            _msg(
                'c',
                'b',
                3,
                role='toolResult',
                toolCallId='t1',
                isError=True,
                content=[{'type': 'text', 'text': 'boom'}],
            ),
        ],
    )
    outputs = [item['output'] for item in parse_file(READER, path).items if item['type'] == 'function_call_output']
    assert outputs == ['Error: boom']


def test_bash_execution_respects_exclude_from_context(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'hi'),
            _msg('b', 'a', 2, role='bashExecution', command='ls', output='x.py', excludeFromContext=False),
            _msg('c', 'b', 3, role='bashExecution', command='secret', output='no', excludeFromContext=True),
        ],
    )
    assert _texts(parse_file(READER, path).items) == ['hi', '$ ls\nx.py']


def test_compaction_becomes_a_system_step_and_preserve_data_is_never_read(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'go'),
            _entry('b', 'a', 2, type='compaction', summary='so far', preserveData={'marker': 'PRESERVE-MARKER-XYZ'}),
        ],
    )
    parsed = parse_file(READER, path)
    assert 'so far' in str(parsed.items[-1])
    assert 'PRESERVE-MARKER-XYZ' not in str(parsed.items)
    document = session_document(parsed)
    assert any(step.source == 'system' and 'so far' in str(step.message) for step in document.trajectory.steps)


def test_custom_messages_map_by_type_and_display(omp_sessions: Path) -> None:
    def custom(record_id: str, parent: str, custom_type: str, content: Any, *, display: bool) -> dict[str, Any]:
        return _entry(
            record_id, parent, 2, type='custom_message', customType=custom_type, content=content, display=display
        )

    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'go'),
            custom('b', 'a', 'skill-prompt', [{'type': 'text', 'text': 'run the skill'}], display=True),
            custom('c', 'b', 'async-result', 'job finished', display=True),
            custom('d', 'c', 'nudge', 'be careful', display=False),
            custom('e', 'd', 'workflow-result', '   ', display=True),
        ],
    )
    parsed = parse_file(READER, path)
    assert [(item['role'], text) for item, text in zip(parsed.items, _texts(parsed.items), strict=True)] == [
        ('user', 'go'),
        ('user', 'run the skill'),
        ('system', '[async-result]\njob finished'),
    ]


def test_reset_boundary_becomes_a_system_step(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'go'),
            _entry('b', 'a', 2, type='reset_boundary'),
            _user('c', 'b', 3, 'again'),
        ],
    )
    parsed = parse_file(READER, path)
    assert _texts(parsed.items) == ['go', 'Conversation context was reset.', 'again']
    assert parsed.items[1]['role'] == 'system'


def test_known_dropped_types_are_silent_and_unknown_types_are_counted(omp_sessions: Path) -> None:
    path = _session(
        omp_sessions,
        [
            *_header(),
            _user('a', None, 1, 'go'),
            _entry('b', 'a', 2, type='custom', customType='tool_execution_start'),
            _entry('c', 'b', 2, type='model_usage'),
            _entry('d', 'c', 2, type='mode_change'),
            _entry('e', 'd', 2, type='ttsr_injection'),
            _msg('f', 'e', 2, role='developer', content='policy'),
            _entry('g', 'f', 2, type='brand_new_type'),
        ],
    )
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING')
    try:
        parsed = parse_file(READER, path)
    finally:
        logger.remove(sink)
    assert _texts(parsed.items) == ['go']
    assert len(messages) == 1
    assert 'brand_new_type' in messages[0]
    assert 'model_usage' not in messages[0]
