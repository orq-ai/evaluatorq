"""Codex reader."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

from evaluatorq.local_sessions import SessionLoadError, session_document
from evaluatorq.local_sessions.codex import READER

from .conftest import write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

SID = '019a0000-0000-7000-8000-000000000001'
NAME = f'rollout-2026-10-01T10-00-00-{SID}.jsonl'


def _ts(minute: int) -> str:
    return f'2026-10-01T10:{minute:02d}:00.000Z'


def _meta(**extra: Any) -> dict[str, Any]:
    payload = {'id': SID, 'timestamp': _ts(0), 'cwd': '/work/proj', 'cli_version': '0.50.0', **extra}
    return {'timestamp': _ts(0), 'type': 'session_meta', 'payload': payload}


def _item(minute: int, **payload: Any) -> dict[str, Any]:
    return {'timestamp': _ts(minute), 'type': 'response_item', 'payload': payload}


def _message(minute: int, role: str, text: str) -> dict[str, Any]:
    kind = 'output_text' if role == 'assistant' else 'input_text'
    return _item(minute, type='message', role=role, content=[{'type': kind, 'text': text}])


def _rollout(home: Path, *, name: str = NAME, records: list[dict[str, Any]] | None = None) -> Path:
    path = home / 'sessions' / '2026' / '10' / '01' / name
    return write_jsonl(path, records if records is not None else [_meta(), _message(1, 'user', 'fix the bug')])


def test_summary_fields_and_index_title(codex_home: Path) -> None:
    path = _rollout(codex_home, records=[_meta(), _message(1, 'user', 'fix the bug'), _message(5, 'assistant', 'done')])
    write_jsonl(
        codex_home / 'session_index.jsonl',
        [
            {'id': SID, 'thread_name': 'old', 'updated_at': _ts(2)},
            {'id': 'other', 'thread_name': 'unrelated', 'updated_at': _ts(2)},
            {'id': SID, 'thread_name': 'Fix the bug', 'updated_at': _ts(3)},
        ],
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.source == 'codex'
    assert summary.session_id == SID
    assert summary.title == 'Fix the bug'
    assert summary.project_dir == '/work/proj'
    assert summary.agent_version == '0.50.0'
    assert summary.first_prompt == 'fix the bug'
    assert summary.started_at.isoformat() == '2026-10-01T10:00:00+00:00'
    assert summary.updated_at.isoformat() == '2026-10-01T10:05:00+00:00'


def test_missing_index_gives_empty_title(codex_home: Path) -> None:
    summary = READER.summarize(_rollout(codex_home))
    assert summary is not None
    assert summary.title == ''


def test_subagent_source_is_not_a_main_session(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[_meta(source={'subagent': {'thread_spawn': {'depth': 1}}}), _message(1, 'user', 'work')],
    )
    assert READER.summarize(path) is None


def test_parent_thread_id_is_not_a_main_session(codex_home: Path) -> None:
    path = _rollout(codex_home, records=[_meta(parent_thread_id='abc'), _message(1, 'user', 'work')])
    assert READER.summarize(path) is None


def test_forked_rollout_is_kept(codex_home: Path) -> None:
    path = _rollout(codex_home, records=[_meta(forked_from_id='abc'), _message(1, 'user', 'work')])
    assert READER.summarize(path) is not None


def test_first_record_must_be_session_meta(codex_home: Path) -> None:
    assert READER.summarize(_rollout(codex_home, records=[_message(1, 'user', 'hi')])) is None


def test_archived_rollout_is_listed(codex_home: Path) -> None:
    archived = codex_home / 'archived_sessions'
    path = write_jsonl(archived / NAME, [_meta(), _message(1, 'user', 'hi')])
    dated = _rollout(codex_home)
    assert list(READER.iter_files(archived)) == [path]
    assert list(READER.iter_files(codex_home / 'sessions')) == [dated]
    assert READER.is_session_path(path, archived)
    assert not READER.is_session_path(archived / 'x' / NAME, archived)
    assert not READER.is_session_path(archived / 'notes.jsonl', archived)


def test_injected_context_is_not_the_first_prompt(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'user', '<environment_context>\n<cwd>/x</cwd>\n</environment_context>'),
            _message(2, 'user', 'real question'),
        ],
    )
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.first_prompt == 'real question'
    parsed = READER.parse(path)
    assert [item['content'][0]['text'] for item in parsed.items] == ['real question']


def test_main_session_without_a_prompt_in_the_head_is_listed(codex_home: Path) -> None:
    path = _rollout(codex_home, records=[_meta(), _message(1, 'user', '<environment_context>x</environment_context>')])
    summary = READER.summarize(path)
    assert summary is not None
    assert summary.first_prompt == ''
    with pytest.raises(SessionLoadError):
        READER.parse(path)


def test_agent_message_becomes_system_step_without_warning(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'user', 'delegate'),
            _item(
                2,
                type='agent_message',
                author='/root/reviewer',
                recipient='/root',
                content=[
                    {'type': 'input_text', 'text': 'found two bugs'},
                    {'type': 'encrypted_content', 'encrypted_content': 'zzz'},
                ],
                id='m1',
                internal_chat_message_metadata_passthrough={},
            ),
            _item(3, type='agent_message', author='/root/empty', recipient='/root', content=[]),
            _message(4, 'assistant', 'thanks'),
        ],
    )
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING')
    try:
        parsed = READER.parse(path)
    finally:
        logger.remove(sink)
    assert not any('agent_message' in message for message in messages)
    system = [item for item in parsed.items if item.get('role') == 'system']
    assert len(system) == 1
    assert system[0]['content'][0]['text'] == 'Message from subagent /root/reviewer:\nfound two bugs'
    assert 'zzz' not in str(parsed.items)
    document = session_document(parsed)
    assert any(step.source == 'system' and 'found two bugs' in str(step.message) for step in document.trajectory.steps)


def test_custom_tool_call_becomes_tool_call_with_observation(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'user', 'patch it'),
            _item(2, type='custom_tool_call', call_id='c1', name='apply_patch', input='*** Begin Patch'),
            _item(3, type='custom_tool_call_output', call_id='c1', output='Success'),
            _item(4, type='function_call', call_id='c2', name='shell', arguments='{"cmd": "ls"}'),
            _item(5, type='function_call_output', call_id='c2', output={'content': 'a.py', 'success': True}),
            _message(6, 'assistant', 'patched'),
        ],
    )
    document = session_document(READER.parse(path))
    calls = [call for step in document.trajectory.steps for call in step.tool_calls or []]
    assert [call.function_name for call in calls] == ['apply_patch', 'shell']
    assert calls[0].arguments == {'input': '*** Begin Patch'}
    observations = [
        result.content for step in document.trajectory.steps if step.observation for result in step.observation.results
    ]
    assert 'Success' in observations
    assert any('a.py' in (content or '') for content in observations)


def test_encrypted_only_reasoning_is_dropped_and_summary_kept(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'user', 'think'),
            _item(2, type='reasoning', summary=[], encrypted_content='xxx'),
            _item(3, type='reasoning', summary=[{'type': 'summary_text', 'text': 'plan A'}], encrypted_content='yyy'),
            _message(4, 'assistant', 'ok'),
        ],
    )
    parsed = READER.parse(path)
    reasoning = [item for item in parsed.items if item['type'] == 'reasoning']
    assert len(reasoning) == 1
    assert reasoning[0]['summary'][0]['text'] == 'plan A'
    assert 'xxx' not in str(parsed.items)


def test_developer_messages_are_dropped_and_compaction_kept(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'developer', 'secret policy'),
            _message(2, 'user', 'go'),
            {'timestamp': _ts(3), 'type': 'compacted', 'payload': {'message': 'so far'}},
        ],
    )
    parsed = READER.parse(path)
    assert 'secret policy' not in str(parsed.items)
    assert 'so far' in str(parsed.items[-1])


def test_model_and_total_tokens_come_from_last_records(codex_home: Path) -> None:
    def tokens(minute: int, total: int) -> dict[str, Any]:
        info = {'total_token_usage': {'total_tokens': total}}
        return {'timestamp': _ts(minute), 'type': 'event_msg', 'payload': {'type': 'token_count', 'info': info}}

    path = _rollout(
        codex_home,
        records=[
            _meta(),
            {'timestamp': _ts(1), 'type': 'turn_context', 'payload': {'model': 'gpt-a'}},
            _message(1, 'user', 'go'),
            tokens(2, 100),
            {'timestamp': _ts(3), 'type': 'turn_context', 'payload': {'model': 'gpt-b'}},
            tokens(4, 250),
            {'timestamp': _ts(5), 'type': 'event_msg', 'payload': {'type': 'token_count', 'info': None}},
            _message(6, 'assistant', 'done'),
        ],
    )
    parsed = READER.parse(path)
    assert parsed.total_tokens == 250
    assert parsed.model == 'gpt-b'


def test_unknown_item_type_warns_once_with_type_and_path(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'user', 'go'),
            _item(2, type='web_search_call', status='completed'),
            _item(3, type='web_search_call', status='completed'),
        ],
    )
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING')
    try:
        READER.parse(path)
    finally:
        logger.remove(sink)
    warnings = [message for message in messages if 'Skipped Codex item types' in message]
    assert len(warnings) == 1
    assert 'web_search_call' in warnings[0]
    assert str(path) in warnings[0]


def test_unreadable_session_raises_load_error_without_content(codex_home: Path) -> None:
    missing = codex_home / 'sessions' / '2026' / '10' / '01' / NAME
    with pytest.raises(SessionLoadError) as caught:
        READER.summarize(missing)
    assert 'FileNotFoundError' in str(caught.value)


def test_long_custom_tool_input_gives_no_argument_warning(codex_home: Path) -> None:
    path = _rollout(
        codex_home,
        records=[
            _meta(),
            _message(1, 'user', 'patch it'),
            _item(2, type='custom_tool_call', call_id='c1', name='exec', input='y' * 25_000),
            _item(3, type='custom_tool_call_output', call_id='c1', output='ok'),
        ],
    )
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING')
    try:
        document = session_document(READER.parse(path))
    finally:
        logger.remove(sink)
    assert not any('not a JSON object' in message for message in messages)
    calls = [call for step in document.trajectory.steps for call in step.tool_calls or []]
    assert calls[0].arguments['input'].endswith('[truncated 5000 chars]')
