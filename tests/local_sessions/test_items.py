"""Shared Responses item builders."""

# ruff: noqa: S101

from __future__ import annotations

from evaluatorq.formats import ResponsesConversation
from evaluatorq.local_sessions import items
from evaluatorq.local_sessions.models import MAX_TOOL_TEXT_CHARS


def test_trim_boundary() -> None:
    exact = 'x' * MAX_TOOL_TEXT_CHARS
    assert items.trim(exact) == exact
    cut = items.trim(exact + 'y')
    assert cut == exact + '\n[truncated 1 chars]'


def test_blocks_text_mixed_blocks() -> None:
    blocks = [
        {'type': 'text', 'text': 'a'},
        {'type': 'image', 'source': {}},
        {'type': 'tool_reference'},
        'junk',
        {'type': 'output_text', 'text': 'b'},
    ]
    assert items.blocks_text(blocks) == 'a\n[image]\n[tool_reference]\nb'
    assert items.blocks_text('plain') == 'plain'
    assert items.blocks_text(None) == ''


def test_user_and_assistant_items_become_user_and_agent_steps() -> None:
    trajectory = ResponsesConversation(items=[items.user_text('hi'), items.assistant_text('yo')]).to_atif(
        agent_name='t', agent_version='1', session_id='s'
    )
    assert [step.source for step in trajectory.steps] == ['user', 'agent']
    assert trajectory.steps[1].message == 'yo'


def test_function_call_serialises_arguments_and_trims() -> None:
    call = items.function_call(call_id='c', name='n', arguments={'k': 'é'})
    assert call['arguments'] == '{"k": "é"}'
    assert items.function_call(call_id='c', name='n', arguments=None)['arguments'] == '{}'
    assert items.function_call_output(call_id='c', output='z' * 25_000)['output'].endswith('[truncated 5000 chars]')
