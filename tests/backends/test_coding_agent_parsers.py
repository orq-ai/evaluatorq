"""Pure parsers over captured JSONL runs. No subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.backends.coding_agent import parse_events
from evaluatorq.openresponses.convert_models import FunctionCallStatus

FIXTURES = Path(__file__).parent / 'fixtures'


def _events(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (FIXTURES / f'{name}.jsonl').read_text().splitlines() if line.strip()]


def test_claude_tool_turn() -> None:
    turn = parse_events('claude', _events('claude_tool'))
    assert turn.text == 'done'
    assert turn.session_id == '9c0bf077-5b29-4b47-9953-3dd93ce096c9'
    assert turn.cost_usd == pytest.approx(0.3873005)
    assert turn.agent_error is None
    [call] = turn.tool_calls
    assert call.name == 'Bash'
    assert json.loads(call.arguments)['command'] == 'echo hello-fixture'
    assert call.result == 'hello-fixture'
    assert call.status is FunctionCallStatus.completed
    assert turn.usage is not None
    assert turn.usage.output_tokens == 83
    assert turn.usage.cached_tokens == 55631
    assert turn.usage.cache_creation_tokens == 35642
    assert turn.model == 'claude-opus-5'  # last assistant message's model, not modelUsage's first key


def test_claude_denied_turn_has_text_and_no_tool_calls() -> None:
    turn = parse_events('claude', _events('claude_denied'))
    assert turn.text
    assert turn.tool_calls == []


def test_claude_permission_denial_becomes_denied_tool_call() -> None:
    events = [
        {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'id': 't1', 'name': 'Bash', 'input': {'command': 'rm -rf /'}}]}},
        {
            'type': 'result',
            'subtype': 'success',
            'is_error': False,
            'result': 'I was not allowed to.',
            'session_id': 's1',
            'usage': {'input_tokens': 1, 'output_tokens': 2},
            'permission_denials': [{'tool_name': 'Bash', 'tool_use_id': 't1', 'tool_input': {'command': 'rm -rf /'}}],
        },
    ]
    turn = parse_events('claude', events)
    [call] = turn.tool_calls
    assert call.result == '[denied by claude]'
    assert call.status is FunctionCallStatus.incomplete


def test_claude_is_error_sets_agent_error() -> None:
    events = [{'type': 'result', 'subtype': 'error_during_execution', 'is_error': True, 'result': 'boom', 'session_id': 's'}]
    assert parse_events('claude', events).agent_error == 'boom'


def test_claude_result_with_incomplete_usage_is_none() -> None:
    events = [{'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'done', 'usage': {'output_tokens': 2}}]
    assert parse_events('claude', events).usage is None


def test_codex_tool_turn() -> None:
    turn = parse_events('codex', _events('codex_tool'))
    assert turn.text == 'done'
    assert turn.session_id == '01a0c596-5def-7473-9c71-4b7067f350b7'
    assert turn.agent_error is None  # the two 'error' items are warnings, not failures
    [call] = turn.tool_calls
    assert call.name == 'shell'
    assert json.loads(call.arguments)['command'] == "/bin/zsh -lc 'echo hello-fixture'"
    assert call.result == 'hello-fixture\n'
    assert call.status is FunctionCallStatus.completed
    assert turn.usage is not None
    assert turn.usage.input_tokens == 40340
    assert turn.usage.cached_tokens == 19968
    assert turn.usage.reasoning_tokens == 78


def test_codex_denied_turn_is_text_only() -> None:
    turn = parse_events('codex', _events('codex_denied'))
    assert turn.text == 'blocked'
    assert turn.tool_calls == []


def test_codex_turn_failed_sets_agent_error() -> None:
    events = [{'type': 'thread.started', 'thread_id': 't'}, {'type': 'turn.failed', 'error': {'message': 'quota'}}]
    assert parse_events('codex', events).agent_error == 'quota'


def test_codex_missing_primary_usage_is_none() -> None:
    events = [{'type': 'turn.completed', 'usage': {'output_tokens': 5}}]
    assert parse_events('codex', events).usage is None


def test_codex_file_change_and_mcp_items() -> None:
    events = [
        {'type': 'item.completed', 'item': {'id': 'i1', 'type': 'file_change', 'status': 'completed', 'changes': [{'path': 'a.py', 'kind': 'add'}]}},
        {'type': 'item.completed', 'item': {'id': 'i2', 'type': 'mcp_tool_call', 'status': 'failed', 'server': 'orq', 'tool': 'list_prompts', 'arguments': {}, 'result': None, 'error': {'message': 'nope'}}},
        {'type': 'item.completed', 'item': {'id': 'i3', 'type': 'agent_message', 'text': 'ok'}},
    ]
    turn = parse_events('codex', events)
    names = [c.name for c in turn.tool_calls]
    assert names == ['apply_patch', 'orq.list_prompts']
    assert turn.tool_calls[1].status is FunctionCallStatus.incomplete
    assert turn.tool_calls[1].result == 'nope'


def test_opencode_tool_turn() -> None:
    turn = parse_events('opencode', _events('opencode_tool'))
    assert turn.text == 'done'
    assert turn.session_id == 'ses_f3a6a0deeffef3YgWSIzjeWzMI'
    [call] = turn.tool_calls
    assert call.name == 'bash'
    assert call.call_id == 'call_YZjixfB8dfHuanC8hmSs5RSa'
    assert json.loads(call.arguments)['command'] == 'echo hello-fixture'
    assert call.result == 'hello-fixture\n'
    assert call.status is FunctionCallStatus.completed
    assert turn.usage is not None
    assert turn.usage.input_tokens == 58843 + 1054
    assert turn.usage.calls == 2
    assert turn.usage.output_tokens == 33 + 5
    assert turn.usage.cached_tokens == 57856
    assert turn.usage.reasoning_tokens == 19
    assert turn.usage.total_tokens == turn.usage.input_tokens + turn.usage.output_tokens  # opencode's own total adds cache reads
    assert turn.cost_usd == 0


def test_opencode_text_fixture_parses() -> None:
    turn = parse_events('opencode', _events('opencode_text'))
    assert turn.text
    assert turn.agent_error is None


def test_opencode_uses_last_text_event() -> None:
    events = [
        {'type': 'text', 'part': {'text': 'first'}},
        {'type': 'text', 'part': {'text': 'second'}},
    ]
    assert parse_events('opencode', events).text == 'second'


def test_opencode_structured_tool_error_is_text() -> None:
    events = [
        {
            'type': 'tool_use',
            'part': {
                'callID': 'call-error',
                'tool': 'bash',
                'state': {'status': 'error', 'error': {'message': 'command failed', 'code': 2}},
            },
        }
    ]
    [call] = parse_events('opencode', events).tool_calls
    assert call.result == '{"message": "command failed", "code": 2}'


def test_opencode_missing_cost_is_none() -> None:
    events = [{'type': 'step_finish', 'part': {'tokens': {'input': 3, 'output': 2}}}]
    turn = parse_events('opencode', events)
    assert turn.cost_usd is None


def test_unknown_event_types_are_skipped() -> None:
    turn = parse_events('codex', [{'type': 'something.new'}, {'type': 'item.completed', 'item': {'id': 'x', 'type': 'agent_message', 'text': 'hi'}}])
    assert turn.text == 'hi'


def test_missing_usage_is_none() -> None:
    turn = parse_events('codex', [{'type': 'item.completed', 'item': {'id': 'x', 'type': 'agent_message', 'text': 'hi'}}])
    assert turn.usage is None


def test_no_final_message_is_none_text() -> None:
    assert parse_events('claude', [{'type': 'system', 'subtype': 'init'}]).text is None
