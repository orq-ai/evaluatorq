import logging

from evaluatorq.common.messages import UNKNOWN_ARGUMENTS, drop_unreplayable_tool_records, messages_to_text
from evaluatorq.contracts import AgentResponse, TextOutputItem, ToolCallOutputItem
from evaluatorq.openresponses.input_items import messages_to_responses_input
from evaluatorq.simulation.contracts import Message
from evaluatorq.simulation.runner.simulation import _invert_roles_for_simulator, build_assistant_message


def test_assistant_message_keeps_tool_calls():
    response = AgentResponse(
        output=[
            TextOutputItem(text='Checking that now.', annotations=[]),
            ToolCallOutputItem(
                name='refund', arguments='{}', id='fc_refund_1', call_id='call_refund_1', result='approved'
            ),
        ]
    )
    messages = build_assistant_message(response)
    assistant, tool = messages
    assert assistant.content == 'Checking that now.'
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].function.name == 'refund'
    assert assistant.tool_calls[0].function.arguments == '{}'
    assert tool == Message(role='tool', tool_call_id='call_refund_1', name='refund', content='approved')

    responses_input = messages_to_responses_input(messages)
    tool_output = next(item for item in responses_input if item.get('type') == 'function_call_output')
    assert tool_output == {
        'type': 'function_call_output',
        'call_id': 'call_refund_1',
        'output': 'approved',
    }


def test_tool_result_is_paired_in_chat_completion_order():
    response = AgentResponse(
        output=[
            ToolCallOutputItem(
                name='refund', arguments='{}', id='fc_refund_1', call_id='call_refund_1', result='approved'
            )
        ]
    )

    rendered = [message.to_chat_completion() for message in build_assistant_message(response)]

    assert rendered[0]['role'] == 'assistant'
    assert rendered[0]['tool_calls'][0]['id'] == 'call_refund_1'
    assert rendered[1] == {
        'role': 'tool',
        'tool_call_id': 'call_refund_1',
        'name': 'refund',
        'content': 'approved',
    }


def _incomplete_response():
    return AgentResponse(
        output=[
            TextOutputItem(text='Checking.', annotations=[]),
            ToolCallOutputItem(name='refund', arguments='{}', id='fc_refund_1', call_id='call_refund_1'),
            ToolCallOutputItem(name='lookup', arguments='{}', id='fc_2', call_id='call_2', result='ok'),
        ]
    )


def test_tool_call_without_result_stays_in_the_transcript_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.runner.simulation'):
        messages = build_assistant_message(_incomplete_response())

    assistant, *tool_messages = messages
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == ['call_refund_1', 'call_2']
    assert [m.tool_call_id for m in tool_messages] == ['call_2']
    assert any('call_refund_1' in r.message and 'no result' in r.message.lower() for r in caplog.records)


def test_judge_text_marks_the_incomplete_call():
    text = messages_to_text(build_assistant_message(_incomplete_response()))

    assert '[tool_call: refund({}) -> no result recorded]' in text
    assert '[tool_call: lookup({})]' in text


def test_judge_text_marks_incomplete_call_with_unknown_arguments():
    response = AgentResponse(output=[ToolCallOutputItem(name='refund', arguments=UNKNOWN_ARGUMENTS, call_id='c1')])

    assert '[tool_call: refund(arguments unavailable) -> no result recorded]' in messages_to_text(
        build_assistant_message(response)
    )


def test_replay_drops_unpaired_call_and_orphan_result_with_one_warning(caplog):
    messages = [
        *build_assistant_message(_incomplete_response()),
        Message(role='tool', tool_call_id='call_orphan', content='stray'),
    ]

    with caplog.at_level(logging.WARNING):
        replay = drop_unreplayable_tool_records(messages)
        items = messages_to_responses_input(replay)

    assert [i['call_id'] for i in items if i.get('type') == 'function_call'] == ['call_2']
    assert [i['call_id'] for i in items if i.get('type') == 'function_call_output'] == ['call_2']
    warnings = [r.message for r in caplog.records if 'Not replaying' in r.message]
    assert len(warnings) == 1
    assert 'call_refund_1' in warnings[0] and 'call_orphan' in warnings[0] and 'call_2' not in warnings[0]


def test_replay_keeps_a_complete_pair_untouched(caplog):
    response = AgentResponse(
        output=[ToolCallOutputItem(name='lookup', arguments='{}', id='fc_2', call_id='call_2', result='ok')]
    )
    messages = build_assistant_message(response)

    with caplog.at_level(logging.WARNING):
        assert drop_unreplayable_tool_records(messages) == messages

    assert not [r for r in caplog.records if 'Not replaying' in r.message]


def test_replay_keeps_a_paired_unknown_arguments_call_with_empty_arguments():
    messages = build_assistant_message(
        AgentResponse(output=[ToolCallOutputItem(name='x', arguments=UNKNOWN_ARGUMENTS, call_id='c1', result='ok')])
    )

    replay = drop_unreplayable_tool_records(messages)
    items = messages_to_responses_input(replay)

    assert replay == messages
    assert [i['arguments'] for i in items if i.get('type') == 'function_call'] == ['']


def test_replay_keeps_text_of_an_assistant_turn_whose_only_call_is_dropped():
    replay = drop_unreplayable_tool_records(build_assistant_message(_incomplete_response())[:1])

    assert [(m.content, m.tool_calls) for m in replay] == [('Checking.', None)]


def test_assistant_text_and_tool_calls_are_kept_together():
    response = AgentResponse(
        output=[
            TextOutputItem(text='Let me check.', annotations=[]),
            ToolCallOutputItem(
                name='lookup', arguments='{"city":"Amsterdam"}', id='fc_1', call_id='call_1', result='sunny'
            ),
        ]
    )

    assistant = build_assistant_message(response)[0]

    assert assistant.content == 'Let me check.'
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].id == 'call_1'


def test_tool_only_turn_warns(caplog):
    response = AgentResponse(
        output=[ToolCallOutputItem(name='refund', arguments='{}', result='approved')]
    )
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.runner.simulation'):
        build_assistant_message(response)
    assert any('tool call' in r.message.lower() for r in caplog.records)


def test_assistant_message_with_text_and_no_tool_calls_does_not_warn(caplog):
    response = AgentResponse(output=[TextOutputItem(text='hello', annotations=[])])
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.runner.simulation'):
        msg = build_assistant_message(response)[0]
    assert msg.content == 'hello'
    assert msg.tool_calls is None
    assert not any('tool call' in r.message.lower() for r in caplog.records)


def test_simulator_view_drops_tool_calls_and_tool_rows():
    """Inverted roles must not leave a `user` row with tool_calls, nor orphan `tool` rows.

    Providers reject "messages with role 'tool' must be a response to a preceeding
    message with 'tool_calls'", which killed the whole simulation at turn 2.
    """
    response = AgentResponse(
        output=[
            TextOutputItem(text='Your balance is €12.', annotations=[]),
            ToolCallOutputItem(
                name='get_card_info', arguments='{}', id='fc_1', call_id='call_1', result='{"balance": 12}'
            ),
        ]
    )
    transcript = [Message(role='user', content='balance?'), *build_assistant_message(response)]

    inverted = _invert_roles_for_simulator(transcript)

    assert [m.role for m in inverted] == ['assistant', 'user']
    assert all(m.tool_calls is None for m in inverted)
    # The target spoke, so the simulator sees only what a real user would.
    assert inverted[1].content == 'Your balance is €12.'


def test_tool_only_turn_reaches_the_simulator_as_text():
    """A turn that is purely tool calls must not invert to a blank `user` row.

    The structure is still dropped (see the test above); the traffic survives as
    text, or the simulator is asked to reply to nothing.
    """
    response = AgentResponse(
        output=[
            ToolCallOutputItem(
                name='get_card_info', arguments='{"id": 7}', id='fc_1', call_id='call_1', result='{"balance": 12}'
            ),
        ]
    )
    transcript = [Message(role='user', content='balance?'), *build_assistant_message(response)]

    inverted = _invert_roles_for_simulator(transcript)

    assert [m.role for m in inverted] == ['assistant', 'user']
    assert inverted[1].tool_calls is None
    content = inverted[1].content
    assert isinstance(content, str)
    assert 'get_card_info({"id": 7})' in content
    assert '{"balance": 12}' in content


def test_long_tool_result_is_truncated_for_the_simulator():
    response = AgentResponse(
        output=[
            ToolCallOutputItem(name='dump', arguments='{}', id='fc_1', call_id='call_1', result='x' * 5000),
        ]
    )

    inverted = _invert_roles_for_simulator(build_assistant_message(response))

    content = inverted[0].content
    assert isinstance(content, str)
    assert '(truncated)' in content
    assert len(content) < 700


def test_max_tool_result_chars_is_configurable():
    """F7: _MAX_TOOL_RESULT_CHARS was a module constant with no way to raise it
    for a tool-heavy agent. Both the default and an explicit override must apply."""
    response = AgentResponse(
        output=[
            ToolCallOutputItem(name='dump', arguments='{}', id='fc_1', call_id='call_1', result='x' * 5000),
        ]
    )
    messages = build_assistant_message(response)

    default_inverted = _invert_roles_for_simulator(messages)
    default_content = default_inverted[0].content
    assert isinstance(default_content, str)
    assert default_content.count('x') == 500  # module default

    raised_inverted = _invert_roles_for_simulator(messages, max_tool_result_chars=2000)
    raised_content = raised_inverted[0].content
    assert isinstance(raised_content, str)
    assert raised_content.count('x') == 2000
