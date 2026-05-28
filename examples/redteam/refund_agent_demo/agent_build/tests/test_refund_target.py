import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_build.refund_target import RefundAgentTarget


def _mk_response(*, task_id='task_1', pending=None, text=None, usage=None):
    parts = [SimpleNamespace(kind='text', text=text)] if text else []
    output = [SimpleNamespace(parts=parts)] if parts else []
    return SimpleNamespace(
        task_id=task_id,
        pending_tool_calls=pending or [],
        output=output,
        usage=usage,
        finish_reason='stop',
    )


def _mk_tool_call(*, call_id, name, args):
    # PendingToolCalls shape: id, type='function', function={name, arguments}
    return SimpleNamespace(
        id=call_id,
        type='function',
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


@pytest.mark.asyncio
async def test_send_prompt_no_pending_tool_calls():
    mock_client = MagicMock()
    mock_client.agents.responses.create.return_value = _mk_response(text='hello world')

    target = RefundAgentTarget(agent_key='test-agent', orq_client=mock_client)
    result = await target.send_prompt('hi')

    assert result.text == 'hello world'
    assert mock_client.agents.responses.create.call_count == 1


@pytest.mark.asyncio
async def test_send_prompt_handles_lookup_order_call():
    mock_client = MagicMock()
    mock_client.agents.responses.create.side_effect = [
        _mk_response(pending=[_mk_tool_call(call_id='c1', name='lookup_order', args={'order_id': 'ord_a1'})]),
        _mk_response(text='found'),
    ]

    target = RefundAgentTarget(agent_key='test-agent', orq_client=mock_client)
    result = await target.send_prompt('show ord_a1')

    assert result.text == 'found'
    assert mock_client.agents.responses.create.call_count == 2

    second_call_kwargs = mock_client.agents.responses.create.call_args_list[1].kwargs
    assert second_call_kwargs['message']['role'] == 'tool'
    parts = second_call_kwargs['message']['parts']
    assert len(parts) == 1
    assert parts[0]['kind'] == 'tool_result'
    assert parts[0]['tool_call_id'] == 'c1'
    assert parts[0]['result']['ok'] is True


@pytest.mark.asyncio
async def test_send_prompt_handles_foreign_lookup_with_404():
    mock_client = MagicMock()
    mock_client.agents.responses.create.side_effect = [
        _mk_response(pending=[_mk_tool_call(call_id='c1', name='lookup_order', args={'order_id': 'ord_b1'})]),
        _mk_response(text='refused'),
    ]

    target = RefundAgentTarget(agent_key='test-agent', orq_client=mock_client)
    await target.send_prompt('show ord_b1')

    second_call_kwargs = mock_client.agents.responses.create.call_args_list[1].kwargs
    result_payload = second_call_kwargs['message']['parts'][0]['result']
    assert result_payload['ok'] is False
    assert result_payload['status_code'] == 404


@pytest.mark.asyncio
async def test_send_prompt_unknown_tool_returns_error():
    mock_client = MagicMock()
    mock_client.agents.responses.create.side_effect = [
        _mk_response(pending=[_mk_tool_call(call_id='c1', name='enable_audit_mode', args={})]),
        _mk_response(text='ok'),
    ]

    target = RefundAgentTarget(agent_key='test-agent', orq_client=mock_client)
    await target.send_prompt('audit')

    second_call_kwargs = mock_client.agents.responses.create.call_args_list[1].kwargs
    payload = second_call_kwargs['message']['parts'][0]['result']
    assert payload['ok'] is False
    assert 'unknown_tool' in payload['error']


@pytest.mark.asyncio
async def test_send_prompt_raises_on_runaway_tool_calls():
    mock_client = MagicMock()
    mock_client.agents.responses.create.return_value = _mk_response(
        pending=[_mk_tool_call(call_id='c1', name='lookup_order', args={'order_id': 'ord_a1'})]
    )

    target = RefundAgentTarget(agent_key='test-agent', orq_client=mock_client)
    with pytest.raises(RuntimeError, match='too many tool-call continuations'):
        await target.send_prompt('loop')


def test_new_returns_fresh_instance_with_same_agent_key():
    mock_client = MagicMock()
    target = RefundAgentTarget(agent_key='test-agent', orq_client=mock_client)
    fresh = target.new()
    assert fresh.agent_key == 'test-agent'
    assert fresh is not target
    assert fresh._task_id is None
