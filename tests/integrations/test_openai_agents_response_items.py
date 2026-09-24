"""The Agents SDK adapter uses the shared Responses item classifier."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai.types.responses import ResponseCustomToolCall, ResponseCustomToolCallOutput

pytest.importorskip('agents')

from evaluatorq.integrations.openai_agents_integration.target import OpenAIAgentTarget


def test_mcp_and_custom_items_keep_their_calls_and_results() -> None:
    target = OpenAIAgentTarget(MagicMock())
    items = [
        {'type': 'mcp_call', 'id': 'mcp_1', 'name': 'search', 'arguments': '{"q":"x"}', 'output': 'hit'},
        {'type': 'custom_tool_call', 'call_id': 'ct_1', 'name': 'run_sql', 'input': 'SELECT 1'},
        {'type': 'custom_tool_call_output', 'call_id': 'ct_1', 'output': '1'},
    ]

    response = target._build_response(items, SimpleNamespace(final_output='done', context_wrapper=None))

    assert [(call.name, call.call_id, call.result) for call in response.tool_calls] == [
        ('search', 'mcp_1', 'hit'),
        ('run_sql', 'ct_1', '1'),
    ]
    assert response.text == 'done'


def test_sdk_tool_objects_keep_their_call_id() -> None:
    target = OpenAIAgentTarget(MagicMock())
    response = target._build_response(
        [
            ResponseCustomToolCall(type='custom_tool_call', call_id='ct_2', name='run_sql', input='SELECT 2'),
            ResponseCustomToolCallOutput(type='custom_tool_call_output', call_id='ct_2', output='2'),
        ],
        SimpleNamespace(final_output='done', context_wrapper=None),
    )

    assert [(call.call_id, call.result) for call in response.tool_calls] == [('ct_2', '2')]
