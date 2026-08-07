import json

from evaluatorq.common.output_adapters import (
    inputs_to_messages,
    output_error_text,
    output_to_messages,
    output_to_text,
)
from evaluatorq.contracts import (
    AgentResponse,
    AgentResponseError,
    ReasoningOutputItem,
    TextOutputItem,
    ToolCallOutputItem,
)


def test_output_to_text_agentresponse_returns_text():
    assert output_to_text(AgentResponse(text='the answer')) == 'the answer'


def test_output_to_text_str_passthrough():
    assert output_to_text('plain') == 'plain'


def test_output_to_text_none_is_empty():
    assert output_to_text(None) == ''


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


def test_output_error_text_none_for_healthy_outputs():
    assert output_error_text(AgentResponse(text='fine')) is None
    assert output_error_text('plain') is None
    assert output_error_text(None) is None
