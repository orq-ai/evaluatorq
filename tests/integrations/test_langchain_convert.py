"""Tests for LangChain -> OpenResponses conversion (convert_to_open_responses)."""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from evaluatorq.integrations.langchain_integration.convert import convert_to_open_responses


def test_langchain_ai_message_with_text_and_tool_calls_keeps_text():
    """An AIMessage with both prose content and tool_calls must keep the prose.

    Previously the AI-message branch only emitted the output-text message in
    the `else` of `if tool_calls:`, so any assistant text alongside tool calls
    was silently dropped from the converted output.
    """
    msg = AIMessage(
        content='I will look that up.',
        tool_calls=[{'name': 'search', 'args': {}, 'id': 'c1'}],
    )
    result = convert_to_open_responses([msg])
    output = result.get('output', [])

    types = [item['type'] for item in output]
    assert 'function_call' in types

    texts = [item.get('content') for item in output]
    assert any('I will look that up.' in str(t) for t in texts)


def test_dict_messages_convert():
    """messages_to_dict() output must convert, not be skipped as 'unknown'.

    `_get_message_type` previously did `getattr(msg, 'type', None)` on a
    dict, which always returns None, so every dict-form message fell into
    the 'unknown' branch and was dropped.
    """
    msgs = [HumanMessage(content='hi'), AIMessage(content='hello')]
    result = convert_to_open_responses(messages_to_dict(msgs))
    assert result.get('input')
    assert result.get('output')


def test_unhashable_dict_message_type_is_skipped(caplog: pytest.LogCaptureFixture):
    """A malformed unhashable message type is skipped, and the skip is announced.

    The empty ``output`` alone proves nothing — an unrecognised type produces no
    items whether or not the degraded path ran. The warning is what distinguishes
    "skipped this message on purpose" from "crashed before producing anything".
    """
    logger_name = 'evaluatorq.integrations.langchain_integration.convert'
    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = convert_to_open_responses([{'type': ['custom']}])

    assert result.get('output') == []
    assert result.get('input') == []
    assert 'Skipping unknown LangChain message type' in caplog.text
    assert "['custom']" in caplog.text


def test_unhashable_str_subclass_message_type_is_converted(caplog: pytest.LogCaptureFixture):
    """A ``str`` subclass with ``__hash__`` disabled can't be used as a dict key,
    but it still names a known message type. The lookup normalises it with
    ``str(...)``, so it converts exactly like a plain ``'human'`` type instead of
    raising ``TypeError`` or being reported as unrecognised.
    """

    class UnhashableStr(str):
        __hash__ = None  # pyright: ignore[reportAssignmentType]

    logger_name = 'evaluatorq.integrations.langchain_integration.convert'
    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = convert_to_open_responses([{'type': UnhashableStr('human'), 'data': {'content': 'hi'}}])

    plain = convert_to_open_responses([{'type': 'human', 'data': {'content': 'hi'}}])

    def _without_ids(items: object) -> list[dict[str, object]]:
        # Message ids are random per call, so compare everything else.
        return [{k: v for k, v in item.items() if k != 'id'} for item in items or []]  # pyright: ignore[reportAttributeAccessIssue, reportGeneralTypeIssues]

    assert _without_ids(result.get('input')) == _without_ids(plain.get('input'))
    assert result.get('input')
    assert 'Skipping unknown LangChain message type' not in caplog.text
