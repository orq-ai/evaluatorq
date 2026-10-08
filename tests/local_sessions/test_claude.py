"""Claude Code and Claude desktop reader."""

# ruff: noqa: S101

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

from evaluatorq.local_sessions import SessionLoadError, session_document
from evaluatorq.local_sessions.claude import READER
from evaluatorq.local_sessions.models import MAX_TOOL_TEXT_CHARS

from .conftest import parse_file, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

SID = 'sess-1'


def _ts(minute: int) -> str:
    return f'2026-10-01T10:{minute:02d}:00.000Z'


def _rec(kind: str, uuid: str, parent: str | None, minute: int, **extra: Any) -> dict[str, Any]:
    return {
        'type': kind,
        'uuid': uuid,
        'parentUuid': parent,
        'sessionId': SID,
        'cwd': '/work/proj',
        'version': '2.1.0',
        'timestamp': _ts(minute),
        'entrypoint': 'cli',
        **extra,
    }


def _user(uuid: str, parent: str | None, minute: int, content: Any, **extra: Any) -> dict[str, Any]:
    return _rec('user', uuid, parent, minute, message={'role': 'user', 'content': content}, **extra)


def _assistant(
    uuid: str, parent: str | None, minute: int, blocks: list[dict[str, Any]], *, mid: str, **extra: Any
) -> dict[str, Any]:
    message = {
        'id': mid,
        'role': 'assistant',
        'model': 'claude-x',
        'content': blocks,
        'usage': extra.pop('usage', None),
    }
    return _rec('assistant', uuid, parent, minute, message=message, **extra)


def _tool_result(uuid: str, parent: str, minute: int, call_id: str, content: Any) -> dict[str, Any]:
    return _user(uuid, parent, minute, [{'type': 'tool_result', 'tool_use_id': call_id, 'content': content}])


def _text_of(parsed_items: list[dict[str, Any]]) -> str:
    return '\n'.join(str(item) for item in parsed_items)


def test_summary_reads_title_cwd_and_window(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-work-proj' / f'{SID}.jsonl',
        [
            {'type': 'queue-operation', 'operation': 'enqueue'},
            _user('u1', None, 0, 'fix the bug please'),
            _assistant('a1', 'u1', 5, [{'type': 'text', 'text': 'done'}], mid='m1'),
            {'type': 'ai-title', 'aiTitle': 'AI title', 'sessionId': SID},
            {'type': 'custom-title', 'customTitle': 'My title', 'sessionId': SID},
            {'type': 'last-prompt', 'leafUuid': 'a1'},
        ],
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.title == 'My title'
    assert summary.project_dir == '/work/proj'
    assert summary.source == 'claude-code'
    assert summary.session_id == SID
    assert summary.first_prompt == 'fix the bug please'
    assert summary.started_at == datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    assert summary.updated_at == datetime(2026, 10, 1, 10, 5, tzinfo=timezone.utc)
    assert summary.agent_version == '2.1.0'


def test_desktop_and_cowork_map_to_claude_desktop(claude_projects: Path, cowork_root: Path) -> None:
    desktop = write_jsonl(
        claude_projects / '-p' / 'd.jsonl',
        [_user('u1', None, 0, 'hello', entrypoint='claude-desktop')],
    )
    cowork = write_jsonl(
        cowork_root / 'acct' / 'org' / 'local_x' / '.claude' / 'projects' / '-sessions-x' / 'c.jsonl',
        [_user('u1', None, 0, 'hello', entrypoint='local-agent')],
    )
    assert desktop in set(READER.iter_files(claude_projects))
    assert cowork in set(READER.iter_files(cowork_root))
    for path in (desktop, cowork):
        summary = READER.summarize(path)
        assert summary is not None
        assert summary.source == 'claude-desktop'


def test_subagent_and_sidechain_files_are_not_sessions(claude_projects: Path) -> None:
    sub = write_jsonl(claude_projects / '-p' / SID / 'subagents' / 'agent-1.jsonl', [_user('u1', None, 0, 'x')])
    assert not READER.is_session_path(sub, claude_projects)
    assert sub not in set(READER.iter_files(claude_projects))
    side = write_jsonl(claude_projects / '-p' / 'side.jsonl', [_user('u1', None, 0, 'x', isSidechain=True)])
    assert READER.is_session_path(side, claude_projects)
    assert READER.summarize(side) is None


def test_attachment_parents_keep_the_whole_turn(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'a.jsonl',
        [
            _user('u1', None, 0, 'read it'),
            _rec('attachment', 'att', 'u1', 1, attachment={'type': 'x'}),
            _assistant('a1', 'att', 2, [{'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {'f': 1}}], mid='m1'),
            _tool_result('r1', 'a1', 3, 't1', 'file body'),
            _assistant('a2', 'r1', 4, [{'type': 'text', 'text': 'all read'}], mid='m2'),
        ],
    )
    parsed = parse_file(READER, path)
    assert [item['type'] for item in parsed.items] == ['message', 'function_call', 'function_call_output', 'message']


def test_parallel_tool_calls_keep_every_result(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'p.jsonl',
        [
            _user('u1', None, 0, 'two things'),
            _assistant(
                'a1', 'u1', 1, [{'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'c': 'a'}}], mid='m1'
            ),
            _assistant(
                'a2', 'a1', 1, [{'type': 'tool_use', 'id': 't2', 'name': 'Bash', 'input': {'c': 'b'}}], mid='m1'
            ),
            _tool_result('r1', 'a1', 2, 't1', 'out-a'),
            _tool_result('r2', 'a2', 2, 't2', 'out-b'),
            _assistant('a3', 'r2', 3, [{'type': 'text', 'text': 'both done'}], mid='m2'),
        ],
    )
    parsed = parse_file(READER, path)
    outputs = {item['call_id']: item['output'] for item in parsed.items if item['type'] == 'function_call_output'}
    assert outputs == {'t1': 'out-a', 't2': 'out-b'}
    step_observations = [
        result.source_call_id
        for step in session_document(parsed).trajectory.steps
        if step.observation
        for result in step.observation.results
    ]
    assert sorted(str(call) for call in step_observations) == ['t1', 't2']


def _rewind_records(*, leaf: str | None) -> list[dict[str, Any]]:
    records = [
        _user('u1', None, 0, 'first question'),
        _assistant('a1', 'u1', 1, [{'type': 'text', 'text': 'answer'}], mid='m1'),
        _user('u2', 'a1', 2, 'EARLIER prompt'),
        _assistant('a2', 'u2', 3, [{'type': 'text', 'text': 'earlier reply'}], mid='m2'),
        _user('u3', 'a1', 4, 'LATER prompt'),
        _assistant('a3', 'u3', 5, [{'type': 'text', 'text': 'later reply'}], mid='m3'),
    ]
    if leaf is not None:
        records.append({'type': 'last-prompt', 'leafUuid': leaf})
    return records


def test_rewound_branch_is_dropped(claude_projects: Path) -> None:
    path = write_jsonl(claude_projects / '-p' / 'rw.jsonl', _rewind_records(leaf='a3'))
    text = _text_of(parse_file(READER, path).items)
    assert 'first question' in text
    assert 'LATER prompt' in text
    assert 'EARLIER' not in text
    assert 'earlier reply' not in text


def test_leaf_in_earlier_branch_keeps_that_branch(claude_projects: Path) -> None:
    path = write_jsonl(claude_projects / '-p' / 'rw2.jsonl', _rewind_records(leaf='a2'))
    text = _text_of(parse_file(READER, path).items)
    assert 'EARLIER prompt' in text
    assert 'LATER' not in text


def test_without_last_prompt_the_last_uuid_is_the_head(claude_projects: Path) -> None:
    path = write_jsonl(claude_projects / '-p' / 'rw3.jsonl', _rewind_records(leaf=None))
    text = _text_of(parse_file(READER, path).items)
    assert 'LATER prompt' in text
    assert 'EARLIER' not in text


def test_tool_result_with_text_block_is_not_a_prompt(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'tr.jsonl',
        [
            _user('u1', None, 0, 'two things'),
            _assistant('a1', 'u1', 1, [{'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {}}], mid='m1'),
            _assistant('a2', 'a1', 1, [{'type': 'tool_use', 'id': 't2', 'name': 'Bash', 'input': {}}], mid='m1'),
            _user(
                'r1',
                'a1',
                2,
                [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'out-a'}, {'type': 'text', 'text': 'note'}],
            ),
            _tool_result('r2', 'a2', 2, 't2', 'out-b'),
            _assistant('a3', 'r2', 3, [{'type': 'text', 'text': 'both done'}], mid='m2'),
            _user('u2', 'a3', 4, 'next prompt'),
            _assistant('a4', 'u2', 5, [{'type': 'text', 'text': 'next reply'}], mid='m3'),
        ],
    )
    parsed = parse_file(READER, path)
    outputs = {item['call_id'] for item in parsed.items if item['type'] == 'function_call_output'}
    assert outputs == {'t1', 't2'}
    assert 'next reply' in _text_of(parsed.items)


def test_repeated_uuids_emit_once(claude_projects: Path) -> None:
    first = _assistant('a1', 'u1', 1, [{'type': 'text', 'text': 'streamed'}], mid='m1')
    path = write_jsonl(
        claude_projects / '-p' / 'dup.jsonl',
        [
            _user('u1', None, 0, 'hi'),
            first,
            first,
            _assistant('a2', 'a1', 2, [{'type': 'text', 'text': 'end'}], mid='m2'),
        ],
    )
    texts = [item for item in parse_file(READER, path).items if item['role'] == 'assistant']
    assert len(texts) == 2


def test_short_unterminated_session_still_summarizes(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'live.jsonl', [_user('u1', None, 0, 'still writing')], trailing_newline=False
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.first_prompt == 'still writing'
    broken = write_jsonl(
        claude_projects / '-p' / 'live2.jsonl',
        [_user('u1', None, 0, 'hello'), '{"type": "assi'],
        trailing_newline=False,
    )
    assert READER.summarize(broken) is not None


def test_malformed_or_offset_less_timestamps_fall_back_to_the_file_time(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'ts.jsonl',
        [
            _user('u1', None, 0, 'hello', timestamp='not a timestamp'),
            _assistant('a1', 'u1', 1, [{'type': 'text', 'text': 'ok'}], mid='m1', timestamp='2026-10-01T10:01:00'),
        ],
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.first_prompt == 'hello'
    assert summary.started_at.tzinfo is not None
    assert summary.updated_at.tzinfo is not None


def test_compaction_keeps_pre_and_post_turns(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'c.jsonl',
        [
            _user('u1', None, 0, 'before compaction'),
            _assistant('a1', 'u1', 1, [{'type': 'text', 'text': 'pre reply'}], mid='m1'),
            _rec('system', 'cb', None, 2, subtype='compact_boundary', logicalParentUuid='a1'),
            _user('s1', 'cb', 3, 'SUMMARY TEXT', isCompactSummary=True),
            _user('u2', 's1', 4, 'after compaction'),
            _assistant('a2', 'u2', 5, [{'type': 'text', 'text': 'post reply'}], mid='m2'),
        ],
    )
    parsed = parse_file(READER, path)
    text = _text_of(parsed.items)
    assert 'before compaction' in text
    assert 'after compaction' in text
    assert 'post reply' in text
    system = [item for item in parsed.items if item['role'] == 'system']
    assert len(system) == 1
    assert 'SUMMARY TEXT' in system[0]['content'][0]['text']


def test_parse_maps_blocks_to_items(claude_projects: Path) -> None:
    usage = {'input_tokens': 10, 'output_tokens': 5, 'cache_creation_input_tokens': 2, 'cache_read_input_tokens': 3}
    path = write_jsonl(
        claude_projects / '-p' / 'm.jsonl',
        [
            _user('u0', None, 0, 'meta noise', isMeta=True),
            _user('u1', 'u0', 1, 'do it'),
            _assistant('a1', 'u1', 2, [{'type': 'thinking', 'thinking': 'hmm'}], mid='m1', usage=usage),
            _assistant(
                'a2',
                'a1',
                2,
                [{'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'c': 'ls'}}],
                mid='m1',
                usage=usage,
            ),
            _tool_result('r1', 'a2', 3, 't1', [{'type': 'text', 'text': 'listing'}]),
            _assistant('a3', 'r1', 4, [{'type': 'text', 'text': 'finished'}], mid='m2', usage={'output_tokens': 7}),
        ],
    )
    parsed = parse_file(READER, path)
    assert parsed.items == [
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'do it'}]},
        {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'hmm'}]},
        {'type': 'function_call', 'call_id': 't1', 'name': 'Bash', 'arguments': '{"c": "ls"}'},
        {'type': 'function_call_output', 'call_id': 't1', 'output': 'listing'},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'finished'}]},
    ]
    assert parsed.total_tokens == 20 + 7
    assert parsed.model == 'claude-x'
    assert parsed.skipped_lines == 0


def test_long_tool_output_is_trimmed(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'l.jsonl',
        [
            _user('u1', None, 0, 'go'),
            _assistant('a1', 'u1', 1, [{'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {}}], mid='m1'),
            _tool_result('r1', 'a1', 2, 't1', 'z' * 25_000),
        ],
    )
    output = next(item['output'] for item in parse_file(READER, path).items if item['type'] == 'function_call_output')
    half = MAX_TOOL_TEXT_CHARS // 2
    assert output == 'z' * half + '\n[truncated 5000 chars]\n' + 'z' * half


def _capture() -> tuple[list[str], int]:
    seen: list[str] = []
    return seen, logger.add(lambda message: seen.append(str(message)), level='WARNING')


def test_partial_last_line_is_not_malformed(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'pl.jsonl',
        [
            _user('u1', None, 0, 'hi'),
            _assistant('a1', 'u1', 1, [{'type': 'text', 'text': 'yo'}], mid='m1'),
            '{"type": "assis',
        ],
        trailing_newline=False,
    )
    seen, sink = _capture()
    try:
        parsed = parse_file(READER, path)
    finally:
        logger.remove(sink)
    assert parsed.skipped_lines == 0
    assert seen == []


def test_malformed_middle_line_warns(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'mm.jsonl',
        [
            _user('u1', None, 0, 'hi'),
            'NOT-JSON-SECRET-CONTENT',
            _assistant('a1', 'u1', 1, [{'type': 'text', 'text': 'yo'}], mid='m1'),
        ],
    )
    seen, sink = _capture()
    try:
        parsed = parse_file(READER, path)
    finally:
        logger.remove(sink)
    assert parsed.skipped_lines == 1
    assert len(seen) == 1
    assert str(path) in seen[0]
    assert 'SECRET-CONTENT' not in seen[0]


def test_oversized_first_line_does_not_raise(claude_projects: Path) -> None:
    path = write_jsonl(claude_projects / '-p' / 'big.jsonl', ['{"type": "user", "pad": "' + 'x' * 300_000 + '"}'])
    assert READER.summarize(path) is None


def test_first_prompt_beyond_the_head_is_listed_and_parses(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'bigprompt.jsonl',
        [
            _user('u1', None, 1, 'x' * 300_000),
            _assistant('a1', 'u1', 2, [{'type': 'text', 'text': 'ok'}], mid='m1'),
        ],
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.first_prompt == ''
    assert parse_file(READER, path).items


def test_file_with_only_summary_records_is_not_listed(claude_projects: Path) -> None:
    path = write_jsonl(
        claude_projects / '-p' / 'summ.jsonl',
        [{'type': 'summary', 'summary': 'old chat', 'leafUuid': 'x'}, {'type': 'summary', 'summary': 'again'}],
    )
    assert READER.summarize(path) is None
