import json

import pytest

from evaluatorq.common.output_adapters import (
    has_meaningful_output,
    inputs_to_messages,
    output_error_text,
    output_to_messages,
    output_to_text,
)
from evaluatorq.common.messages import messages_to_text
from evaluatorq.contracts import (
    AgentResponse,
    AgentResponseError,
    Message,
    ReasoningOutputItem,
    TextOutputItem,
    ToolCallOutputItem,
)
from evaluatorq.evaluators import exact_match_evaluator
from evaluatorq.types import DataPoint, EvaluationResult


def test_output_to_text_agentresponse_returns_text():
    assert output_to_text(AgentResponse(text='the answer')) == 'the answer'


def test_output_to_text_str_passthrough():
    assert output_to_text('plain') == 'plain'


@pytest.mark.asyncio
async def test_output_to_text_message_list_supports_exact_match():
    output = [Message(role='assistant', content='answer')]

    assert output_to_text(output) == 'answer'

    result = await exact_match_evaluator()['scorer']({
        'data': DataPoint(inputs={}, expected_output='answer'),
        'output': output,
    })
    score = result if isinstance(result, EvaluationResult) else EvaluationResult.model_validate(result)
    assert score.pass_ is True


def test_output_to_text_none_is_empty():
    assert output_to_text(None) == ''


def test_has_meaningful_output_rejects_blank_structured_message():
    assert not has_meaningful_output([{'role': 'assistant', 'content': '   '}])


def test_has_meaningful_output_accepts_tool_call_without_text():
    output = [{'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1'}]}]

    assert has_meaningful_output(output)


def test_has_meaningful_output_accepts_unknown_nonempty_shape():
    assert has_meaningful_output([{'type': 'future_output', 'payload': {}}])


def test_output_to_text_dict_non_response_json():
    assert json.loads(output_to_text({'k': 'v'})) == {'k': 'v'}


def test_output_to_text_never_raises_on_weird():
    class Weird:
        def __str__(self):
            raise RuntimeError('boom')
    # Weird isn't a valid Output, but the helper must be defensive.
    assert isinstance(output_to_text(Weird()), str)  # degrades, no raise


def test_inputs_to_messages_messages_shape():
    msgs = inputs_to_messages({'messages': [{'role': 'user', 'content': 'hi'}]})
    assert msgs == [{'role': 'user', 'content': 'hi'}]


def test_inputs_to_messages_input_shape():
    msgs = inputs_to_messages({'input': 'hello'})
    assert msgs == [{'role': 'user', 'content': 'hello'}]


def test_inputs_to_messages_fallback_json():
    msgs = inputs_to_messages({'foo': 'bar'})
    assert msgs[0]['role'] == 'user' and 'foo' in msgs[0]['content']


def test_output_to_messages_agentresponse_passthrough():
    ar = AgentResponse(text='hi')
    out = output_to_messages(ar)
    assert out == list(ar.output)


def test_output_to_messages_str_wraps_textitem():
    out = output_to_messages('plain')
    assert len(out) == 1 and isinstance(out[0], TextOutputItem) and out[0].text == 'plain'


def test_output_to_messages_none_empty():
    assert output_to_messages(None) == []


def test_output_to_messages_malformed_response_dict_degrades():
    # `output` is a non-iterable int → the from_openresponses item loop raises
    # (TypeError) → output_to_messages catches and degrades to text. A dict that
    # merely lacks fields does NOT raise (from_openresponses returns empty), so it
    # would yield [] — pick an input that genuinely hits the except branch.
    out = output_to_messages({'object': 'response', 'output': 123})
    assert len(out) >= 1  # degraded to a text item, no raise


def test_output_to_messages_reasoning_item_passes_through_without_raise():
    # A reasoning item in AgentResponse.output must not crash the adapter; it is
    # simply carried in output_messages and dropped by _format_output_message /
    # the response-text join downstream (it is neither text nor tool call).
    ar = AgentResponse(output=[ReasoningOutputItem(text='thinking...')])
    out = output_to_messages(ar)
    assert isinstance(out, list)  # no raise; degrades cleanly


def test_output_to_messages_static_dict_with_tool_calls():
    # The plain static-output shape (no `object: 'response'`) that job/DataPoint
    # outputs commonly use — this feeds both juries.
    out = output_to_messages({
        'response': 'done',
        'tool_calls': [{'id': 't1', 'name': 'search', 'arguments': '{"q": "x"}', 'result': 'hit'}],
    })
    assert any(isinstance(i, TextOutputItem) and i.text == 'done' for i in out)
    assert any(isinstance(i, ToolCallOutputItem) and i.name == 'search' for i in out)


def test_output_error_text_reads_agentresponse_error():
    ar = AgentResponse(output=[], error=AgentResponseError(message='timeout', error_type='timeout'))
    assert output_error_text(ar) == 'timeout'


def test_output_error_text_detects_empty_agentresponse_error():
    ar = AgentResponse(output=[], error=AgentResponseError(message='', error_type='target_error'))
    assert output_error_text(ar) == ''
    assert output_error_text(ar) is not None


def test_output_error_text_reads_dict_error_message():
    assert output_error_text({'error': {'message': 'timeout'}}) == 'timeout'


def test_output_error_text_none_for_healthy_outputs():
    assert output_error_text(AgentResponse(text='fine')) is None
    assert output_error_text('plain') is None
    assert output_error_text(None) is None


def test_output_to_text_renders_a_tool_only_turn():
    """has_meaningful_output counts a tool call, so the renderer must too.

    Dropping tool calls here handed the judge an empty string for an agent that
    acted correctly, and scored it as having said nothing.
    """
    recorded = [{'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'get_balance', 'arguments': '{}'}}]}]
    assert has_meaningful_output(recorded) is True
    assert output_to_text(recorded) == '[tool_call: get_balance({})]'


def test_output_to_text_separates_messages_and_labels_non_assistant_turns():
    recorded = [
        {'role': 'tool', 'content': '{"balance": 42}'},
        {'role': 'assistant', 'content': 'You have 42.'},
    ]
    assert output_to_text(recorded) == '[tool] {"balance": 42}\nYou have 42.'


def test_output_to_text_leaves_a_single_assistant_message_bare():
    assert output_to_text([{'role': 'assistant', 'content': 'answer'}]) == 'answer'


def test_output_to_text_preserves_recorded_whitespace():
    """Trimming is for the blank check, not for the text a scorer compares."""
    assert output_to_text([{'role': 'assistant', 'content': ' answer '}]) == ' answer '


def test_output_to_text_renders_a_legacy_function_call():
    """has_meaningful_output counts the legacy fields, so the renderer must too."""
    recorded = [{'role': 'assistant', 'function_call': {'name': 'lookup', 'arguments': '{}'}}]
    assert has_meaningful_output(recorded) is True
    assert output_to_text(recorded) == '[tool_call: lookup({})]'


def test_output_to_text_renders_a_legacy_function_call_output():
    recorded = [{'role': 'assistant', 'function_call_output': {'call_id': 'c1', 'output': '42'}}]
    assert has_meaningful_output(recorded) is True
    assert output_to_text(recorded) == '[tool_result: 42]'


def test_output_to_text_json_renders_a_list_of_structured_items():
    """A Responses output item is a dict but not a message: reading it as one rendered ''."""
    rendered = output_to_text([{'type': 'output_text', 'text': 'answer'}])
    assert 'answer' in rendered


def test_output_to_text_renders_a_tool_call_only_agent_response():
    response = AgentResponse(
        output=[ToolCallOutputItem(id='c1', call_id='c1', name='get_balance', arguments='{}')]
    )
    assert has_meaningful_output(response) is True
    assert output_to_text(response) == '[tool_call: get_balance({})]'


def test_output_to_text_renders_text_and_tool_calls_together():
    """A turn that answered and acted must show both: the action used to vanish behind the text."""
    response = AgentResponse(
        output=[
            TextOutputItem(text='checking that now', annotations=[]),
            ToolCallOutputItem(id='c1', call_id='c1', name='get_balance', arguments='{}'),
        ]
    )
    rendered = output_to_text(response)
    assert 'checking that now' in rendered
    assert '[tool_call: get_balance({})]' in rendered


def test_output_to_text_renders_text_and_tool_calls_in_produced_order():
    response = AgentResponse(
        output=[
            ToolCallOutputItem(id='c1', call_id='c1', name='lookup', arguments='{}'),
            TextOutputItem(text='found it', annotations=[]),
        ]
    )
    rendered = output_to_text(response)
    assert rendered.index('[tool_call: lookup({})]') < rendered.index('found it')


def test_output_to_text_reads_a_list_of_responses_output_items():
    """A Responses message item carries type; reading it as a chat turn rendered '[output_text]'."""
    items = [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'answer'}]},
        {'type': 'function_call', 'name': 'lookup', 'call_id': 'c1', 'arguments': '{}'},
    ]
    rendered = output_to_text(items)
    assert 'answer' in rendered
    assert '[output_text]' not in rendered
    assert '[tool_call: lookup({})]' in rendered


def test_output_to_text_reads_a_list_of_content_parts_as_text():
    assert output_to_text([{'type': 'output_text', 'text': 'answer'}]) == 'answer'
    assert output_to_text([{'type': 'refusal', 'refusal': 'I cannot help with that.'}]) == 'I cannot help with that.'


def test_output_to_text_renders_a_falsey_legacy_tool_result():
    """Presence, not truthiness: has_meaningful_output counts the key, so the text must show it."""
    output = [{'role': 'assistant', 'function_call_output': ''}]
    assert has_meaningful_output(output) is True
    assert output_to_text(output) == '[tool_result: ]'


def test_output_to_text_reads_legacy_fields_from_message_objects():
    class _SdkMessage:
        role = 'assistant'
        content = None
        tool_calls = None
        function_call = {'name': 'get_balance', 'arguments': '{}'}

    assert messages_to_text([_SdkMessage()]) == '[tool_call: get_balance({})]'
