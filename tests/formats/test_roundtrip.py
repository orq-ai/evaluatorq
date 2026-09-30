"""The 4x3 conversion matrix and round-trip guarantees."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.contracts import FunctionCall, Message, StrategyToolCall
from evaluatorq.formats.atif import AtifTrajectory
from evaluatorq.formats.chat import ChatConversation
from evaluatorq.formats.otel import OtelTrace
from evaluatorq.formats.responses import ResponsesConversation

FIX = Path(__file__).parent / 'fixtures'
Format = ChatConversation | ResponsesConversation | OtelTrace | AtifTrajectory


def _chat() -> ChatConversation:
    return ChatConversation(messages=[
        Message(role='user', content='weather?'),
        Message(role='assistant', content=None, tool_calls=[
            StrategyToolCall(id='c1', function=FunctionCall(name='w', arguments='{"city":"Berlin"}'), item_id='fc_1')]),
        Message(role='tool', tool_call_id='c1', name='w', content='12C'),
        Message(role='assistant', content='12C in Berlin'),
    ])


def _all_formats() -> dict[str, Format]:
    chat = _chat()
    otel = OtelTrace.from_orq(json.loads((FIX / 'otel' / 'orq_agent_subagent.json').read_text()))
    atif = AtifTrajectory.from_json((FIX / 'atif' / 'harbor_atif2otel_pass_v17.json').read_text())
    return {'chat': chat, 'responses': chat.to_responses(), 'otel': otel, 'atif': atif}


@pytest.mark.parametrize('source', ['chat', 'responses', 'otel', 'atif'])
@pytest.mark.parametrize('target', ['chat', 'responses', 'otel', 'atif'])
def test_every_pair_converts(source: str, target: str) -> None:
    if source == target:
        pytest.skip('same format')
    obj = _all_formats()[source]
    result = getattr(obj, f'to_{target}')()
    assert type(result).__name__ == {'chat': 'ChatConversation', 'responses': 'ResponsesConversation',
                                    'otel': 'OtelTrace', 'atif': 'AtifTrajectory'}[target]


def test_chat_responses_chat_exact() -> None:
    chat = _chat()
    back = chat.to_responses().to_chat()
    assert [m.model_dump(exclude_none=True) for m in back.messages] == [m.model_dump(exclude_none=True) for m in chat.messages]


def test_chat_atif_chat_keeps_messages_and_tool_linkage() -> None:
    chat = _chat()
    back = chat.to_atif().to_chat()
    assert [m.role for m in back.messages] == [m.role for m in chat.messages]
    assert back.messages[1].tool_calls is not None and back.messages[1].tool_calls[0].id == 'c1'
    assert back.messages[2].tool_call_id == 'c1'


def _step_sig(t: AtifTrajectory) -> list[Any]:
    return [(s.source, bool(s.reasoning_content), [c.tool_call_id for c in s.tool_calls or []]) for s in t.steps]


def test_atif_responses_atif_keeps_messages_calls_reasoning_metrics() -> None:
    src = AtifTrajectory.from_json((FIX / 'atif' / 'harbor_atif2otel_pass_v17.json').read_text())
    back = src.to_responses().to_atif()
    assert _step_sig(back) == _step_sig(src)
    assert [s.message for s in back.steps] == [s.message for s in src.steps]
    assert [s.reasoning_content for s in back.steps] == [s.reasoning_content for s in src.steps]
    def usage(t: AtifTrajectory) -> list[Any]:
        return [(s.metrics.prompt_tokens, s.metrics.completion_tokens) if s.metrics else None for s in t.steps]
    assert usage(back) == usage(src)


def test_atif_responses_atif_merges_steps_whose_results_have_no_call_id() -> None:
    # Documented loss: a result with no source_call_id is dropped (warned), so the agent steps it separated merge,
    # and the per-step responses no longer line up one to one, so their metrics are ignored (warned).
    src = AtifTrajectory.from_json((FIX / 'atif' / 'harbor_invalid_json_v18.json').read_text())
    back = src.to_responses().to_atif()
    assert [s.source for s in back.steps] == ['user', 'agent', 'agent']
    calls = [c.tool_call_id for s in back.steps for c in s.tool_calls or []]
    assert calls == [c.tool_call_id for s in src.steps for c in s.tool_calls or []]
    reasoning = '\n'.join(s.reasoning_content or '' for s in back.steps if s.reasoning_content)
    assert all(s.reasoning_content in reasoning for s in src.steps if s.reasoning_content)
    assert all(s.metrics is None for s in back.steps)


def test_chat_otel_chat_keeps_messages_and_item_ids() -> None:
    chat = _chat()
    back = chat.to_otel().to_chat()
    expected = [m.model_dump(exclude_none=True) for m in chat.messages]
    assert expected[1]['tool_calls'][0]['item_id']  # held in ATIF step extra, carried as a span attribute
    assert [m.model_dump(exclude_none=True) for m in back.messages] == expected


def test_real_orq_trace_survives_atif_otel_atif() -> None:
    raw = json.loads((FIX / 'otel' / 'orq_agent_real.json').read_text())['spans']
    atif = OtelTrace.from_orq(raw).to_atif()
    back = atif.to_otel().to_atif()
    def sig(t: AtifTrajectory) -> list[Any]:
        return [
            (s.source, s.message, s.reasoning_content, [(c.tool_call_id, c.function_name, c.arguments) for c in s.tool_calls or []],
             s.observation, s.metrics, s.llm_call_count)
            for s in t.steps
        ]
    assert [s.source for s in atif.steps] == ['system', 'user', 'agent', 'agent', 'agent']
    assert sig(back) == sig(atif)
    assert back.final_metrics == atif.final_metrics


def test_otel_atif_otel_keeps_span_tree_messages_tool_linkage() -> None:
    otel = _all_formats()['otel']
    assert isinstance(otel, OtelTrace)
    back = otel.to_atif().to_otel()
    ops = sorted(s.operation or '' for s in back.spans)
    assert ops == sorted(['invoke_agent', 'invoke_agent', 'chat', 'chat', 'chat', 'execute_tool'])
    chats = [s for s in back.spans if s.operation == 'chat']
    texts = [m.parts[0].model_dump().get('content') for c in chats for m in (c.output_messages or [])]
    assert 'ATIF is a trajectory format.' in texts


def test_otel_to_chat_composes_through_atif() -> None:
    otel = _all_formats()['otel']
    assert isinstance(otel, OtelTrace)
    assert otel.to_chat() == otel.to_atif().to_responses().to_chat()


def test_reasoning_is_lost_in_chat_but_kept_via_responses() -> None:
    otel = _all_formats()['otel']
    assert isinstance(otel, OtelTrace)
    assert any(i.get('type') == 'reasoning' for i in otel.to_responses().items)
    assert all('delegate.' not in str(m.content) for m in otel.to_chat().messages)


def test_conversion_is_idempotent_across_repeats() -> None:
    otel = _all_formats()['otel']
    assert isinstance(otel, OtelTrace)
    assert otel.to_atif().to_json() == otel.to_atif().to_json()
