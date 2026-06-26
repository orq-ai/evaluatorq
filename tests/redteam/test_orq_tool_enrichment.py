"""Tests for ORQAgentTarget tool enrichment (tool_id -> full schema)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluatorq.redteam.backends.orq import ORQAgentTarget


def _target_with_tools(bindings: list[object], retrieve_return: object = None) -> ORQAgentTarget:
    client = MagicMock()
    client.agents.retrieve.return_value = SimpleNamespace(
        settings=SimpleNamespace(tools=bindings),
        knowledge_bases=None,
        memory_stores=None,
        model=None,
        display_name='a',
        description=None,
        system_prompt=None,
        instructions=None,
    )
    if retrieve_return is not None:
        client.tools.retrieve.return_value = retrieve_return
    return ORQAgentTarget(agent_key='a', orq_client=client)


@pytest.mark.asyncio
async def test_enrich_function_tool_pulls_description_and_parameters():
    params = SimpleNamespace(model_dump=lambda: {'type': 'object', 'properties': {'q': {'type': 'string'}}})
    rich = SimpleNamespace(
        function=SimpleNamespace(description='Search the web', parameters=params),
        json_schema=None,
    )
    binding = SimpleNamespace(
        key=None, display_name='Web Search', id='t1', description=None, action_type='function', tool_id='tool_123'
    )
    target = _target_with_tools([binding], retrieve_return=rich)

    ctx = await target.get_agent_context()

    assert len(ctx.tools) == 1
    tool = ctx.tools[0]
    assert tool.name == 'Web Search'
    assert tool.description == 'Search the web'
    assert tool.parameters == {'type': 'object', 'properties': {'q': {'type': 'string'}}}
    assert tool.action_type == 'function'


@pytest.mark.asyncio
async def test_enrich_json_schema_tool_uses_schema_field():
    schema = SimpleNamespace(model_dump=lambda: {'type': 'object'})
    rich = SimpleNamespace(function=None, json_schema=SimpleNamespace(description='JS tool', schema_=schema))
    binding = SimpleNamespace(
        key='js_key', display_name=None, id='t2', description=None, action_type='json_schema', tool_id='tool_456'
    )
    target = _target_with_tools([binding], retrieve_return=rich)

    ctx = await target.get_agent_context()

    assert ctx.tools[0].name == 'js_key'
    assert ctx.tools[0].description == 'JS tool'
    assert ctx.tools[0].parameters == {'type': 'object'}


@pytest.mark.asyncio
async def test_binding_without_tool_id_degrades_to_binding_fields():
    # Built-in tool (e.g. "Current Date & Time"): no tool_id, no description, no params.
    binding = SimpleNamespace(
        key=None,
        display_name='Current Date & Time',
        id='builtin',
        description=None,
        action_type='datetime',
        tool_id=None,
    )
    target = _target_with_tools([binding])

    ctx = await target.get_agent_context()

    tool = ctx.tools[0]
    assert tool.name == 'Current Date & Time'
    assert tool.description is None
    assert tool.parameters is None
    assert tool.action_type == 'datetime'
    target.orq_client.tools.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_enrich_failure_degrades_gracefully():
    binding = SimpleNamespace(
        key='k', display_name=None, id='t3', description='binding desc', action_type='http', tool_id='tool_err'
    )
    target = _target_with_tools([binding])
    target.orq_client.tools.retrieve.side_effect = RuntimeError('boom')

    ctx = await target.get_agent_context()

    tool = ctx.tools[0]
    assert tool.name == 'k'
    assert tool.description == 'binding desc'  # falls back to binding description
    assert tool.parameters is None
