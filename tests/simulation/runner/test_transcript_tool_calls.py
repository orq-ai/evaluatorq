import logging

import pytest

from evaluatorq.common.thread_context import conversation_thread
from evaluatorq.common.messages import (
    EMPTY_ASSISTANT_TEXT,
    UNKNOWN_ARGUMENTS,
    drop_unreplayable_tool_records,
    messages_to_text,
)
from evaluatorq.contracts import AgentResponse, FunctionCall, StrategyToolCall, TextOutputItem, ToolCallOutputItem
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


def test_tool_call_without_result_stays_in_the_transcript_and_does_not_warn_at_render(caplog):
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.runner.simulation'):
        messages = build_assistant_message(_incomplete_response())

    assistant, *tool_messages = messages
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == ['call_refund_1', 'call_2']
    assert [m.tool_call_id for m in tool_messages] == ['call_2']
    assert assistant.tool_calls[0].incomplete and not assistant.tool_calls[1].incomplete
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and 'call_refund_1' in r.message]


def test_judge_text_marks_the_incomplete_call():
    text = messages_to_text(build_assistant_message(_incomplete_response()))

    assert '[tool_call: refund({}) -> no result recorded]' in text
    assert '[tool_call: lookup({})]' in text


def test_judge_text_marks_incomplete_call_with_unknown_arguments():
    response = AgentResponse(output=[ToolCallOutputItem(name='refund', arguments=UNKNOWN_ARGUMENTS, call_id='c1')])

    assert '[tool_call: refund(arguments unavailable) -> no result recorded]' in messages_to_text(
        build_assistant_message(response)
    )


def _call(call_id: str, args: str = '{}') -> StrategyToolCall:
    return StrategyToolCall(id=call_id, function=FunctionCall(name='f', arguments=args))


def _assistant(*ids: str) -> Message:
    return Message(role='assistant', tool_calls=[_call(i) for i in ids])


def _tool(call_id: str) -> Message:
    return Message(role='tool', tool_call_id=call_id, content='r')


def _user() -> Message:
    return Message(role='user', content='u')


def _system() -> Message:
    return Message(role='system', content='s')


@pytest.mark.parametrize(
    ('label', 'messages', 'expected_roles'),
    [
        ('orphan_result', [_user(), _tool('c1')], ['user']),
        ('result_before_call', [_tool('c1'), _assistant('c1')], ['assistant']),
        ('result_after_later_user_turn', [_assistant('c1'), _user(), _tool('c1')], ['assistant', 'user']),
        ('duplicate_result_keeps_first', [_assistant('c1'), _tool('c1'), _tool('c1')], ['assistant', 'tool']),
        (
            'id_reused_in_a_later_turn',
            [_assistant('c1'), _tool('c1'), _user(), _assistant('c1')],
            ['assistant', 'tool', 'user', 'assistant'],
        ),
        (
            'result_for_another_turns_call',
            [_assistant('c1'), _tool('c1'), _assistant('c2'), _tool('c1')],
            ['assistant', 'tool', 'assistant'],
        ),
        (
            'parallel_calls_split_across_assistant_rows',
            [_assistant('a'), _assistant('b'), _tool('a'), _tool('b')],
            ['assistant', 'tool', 'assistant', 'tool'],
        ),
        (
            'system_row_between_call_and_result',
            [_assistant('a'), _system(), _tool('a')],
            ['assistant', 'tool', 'system'],
        ),
    ],
)
def test_pairing_is_positional(label, messages, expected_roles):
    assert [m.role for m in drop_unreplayable_tool_records(messages)] == expected_roles


@pytest.mark.parametrize(
    ('label', 'messages', 'expected'),
    [
        ('adjacent', [_assistant('a'), _tool('a')], ['A a', 'T a']),
        ('system_between', [_assistant('a'), _system(), _tool('a')], ['A a', 'T a', 'S']),
        ('split_assistant_rows', [_assistant('a'), _assistant('b'), _tool('a'), _tool('b')], ['A a', 'T a', 'A b', 'T b']),
        ('parallel_results_keep_call_order', [_assistant('a', 'b'), _tool('b'), _tool('a')], ['A a,b', 'T a', 'T b']),
        ('other_row_between', [_assistant('a'), _assistant('b'), _tool('b'), _tool('a')], ['A a', 'T a', 'A b', 'T b']),
        ('result_after_later_user_turn', [_assistant('a'), _user(), _tool('a')], ['A -', 'U']),
        ('result_before_call', [_tool('a'), _assistant('a')], ['A -']),
        ('duplicate_result', [_assistant('a'), _tool('a'), _tool('a')], ['A a', 'T a']),
    ],
)
def test_replay_copy_puts_each_result_directly_after_its_call(label, messages, expected):
    def show(m):
        ids = ','.join(tc.id for tc in m.tool_calls or []) or '-'
        return {'assistant': f'A {ids}', 'tool': f'T {m.tool_call_id}', 'system': 'S', 'user': 'U'}[m.role]

    assert [show(m) for m in drop_unreplayable_tool_records(messages)] == expected


def test_orphan_shapes_warn_once(caplog):
    messages = [_assistant('a'), _user(), _tool('a'), _tool('zzz')]
    with conversation_thread('t'), caplog.at_level(logging.WARNING):
        drop_unreplayable_tool_records(messages)
        drop_unreplayable_tool_records(messages)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) <= 1


def test_replay_copy_leaves_the_recorded_transcript_untouched():
    messages = [_assistant('a'), _system(), _tool('a')]
    before = list(messages)

    drop_unreplayable_tool_records(messages)

    assert messages == before


def test_parallel_calls_split_across_rows_keep_their_calls():
    replay = drop_unreplayable_tool_records([_assistant('a'), _assistant('b'), _tool('a'), _tool('b')])

    assert [[tc.id for tc in m.tool_calls or []] for m in replay if m.role == 'assistant'] == [['a'], ['b']]


def test_emptied_assistant_row_keeps_the_alternation_placeholder():
    from evaluatorq.contracts import AgentResponse as _AR
    from evaluatorq.redteam.contracts import Turn, turns_to_messages

    turns = [
        Turn(attacker=_AR(text='one'), target=_AR(output=[ToolCallOutputItem(name='f', arguments='{}', call_id='c1')])),
        Turn(attacker=_AR(text='two'), target=_AR(text='fine')),
    ]
    messages = turns_to_messages(turns)

    replay = drop_unreplayable_tool_records(messages)

    assert [m.role for m in replay] == ['user', 'assistant', 'user', 'assistant']
    assert (replay[1].content, replay[1].tool_calls) == (EMPTY_ASSISTANT_TEXT, None)


def test_whitespace_only_assistant_turn_with_only_dropped_calls_becomes_empty():
    message = Message(role='assistant', content='  \n', tool_calls=[_call('c1')])

    replay = drop_unreplayable_tool_records([_user(), message])

    assert [(m.role, m.content, m.tool_calls) for m in replay] == [
        ('user', 'u', None),
        ('assistant', EMPTY_ASSISTANT_TEXT, None),
    ]


def test_replay_warns_once_per_call_across_turns(caplog):
    messages = [_user(), _assistant('c1')]

    with conversation_thread(), caplog.at_level(logging.DEBUG):
        for _ in range(3):
            drop_unreplayable_tool_records(messages)

    assert [r.levelname for r in caplog.records if 'Not replaying' in r.message] == ['WARNING', 'DEBUG', 'DEBUG']


def test_judge_copy_loses_no_call_so_only_an_orphan_result_warns(caplog):
    messages = [_user(), _assistant('c1'), _tool('stray')]

    with conversation_thread(), caplog.at_level(logging.WARNING):
        drop_unreplayable_tool_records(messages, incomplete_as_text=True)
        drop_unreplayable_tool_records(messages, incomplete_as_text=True)

    warnings = [r.message for r in caplog.records if 'Not replaying' in r.message]
    assert len(warnings) == 1 and 'stray' in warnings[0] and 'c1' not in warnings[0]


def test_rebuilding_the_red_team_transcript_every_turn_warns_once_per_call(caplog):
    from evaluatorq.contracts import AgentResponse as _AR
    from evaluatorq.redteam.contracts import Turn, turns_to_messages

    turns = [
        Turn(attacker=_AR(text='x'), target=_AR(output=[ToolCallOutputItem(name='f', arguments='{}', call_id='c1')]))
    ]

    with conversation_thread(), caplog.at_level(logging.WARNING):
        for _ in range(3):
            drop_unreplayable_tool_records(turns_to_messages(turns))

    assert sum('c1' in r.message for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_simulation_warns_once_per_call_across_turns(caplog):
    with conversation_thread(), caplog.at_level(logging.WARNING):
        for _ in range(3):
            messages = build_assistant_message(_incomplete_response())
            drop_unreplayable_tool_records(messages)

    assert sum('call_refund_1' in r.message for r in caplog.records if r.levelno == logging.WARNING) == 1


def test_tool_traffic_text_shares_the_marker_grammar():
    from evaluatorq.simulation.runner.simulation import _tool_traffic_text

    message = Message(role='assistant', tool_calls=[_call('a'), _call('b')])

    assert _tool_traffic_text(message, {'a': 'ok'}).splitlines()[1:] == [
        '[tool_call: f({}) -> ok]',
        '[tool_call: f({}) -> no result recorded]',
    ]


def test_export_round_trip_keeps_the_marker():
    from evaluatorq.common.output_adapters import responses_transcript
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.convert import to_open_responses
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy

    result = SimulationResult(
        messages=[Message(role='user', content='refund'), *build_assistant_message(_incomplete_response())],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        turn_metrics=[],
    )

    transcript = responses_transcript(to_open_responses(result))

    assert transcript is not None
    text = messages_to_text(transcript)
    assert '[tool_call: refund({}) -> no result recorded]' in text
    assert '[tool_call: lookup({})]' in text


def test_incomplete_calls_become_text_for_the_judge_but_not_for_targets():
    messages = build_assistant_message(_incomplete_response())

    for_judge = drop_unreplayable_tool_records(messages, incomplete_as_text=True)
    for_target = drop_unreplayable_tool_records(messages)

    assert for_judge[0].content == 'Checking.\n[tool_call: refund({}) -> no result recorded]'
    assert for_target[0].content == 'Checking.'
    assert [tc.id for tc in for_judge[0].tool_calls or []] == ['call_2']
    assert [tc.id for tc in for_target[0].tool_calls or []] == ['call_2']


def test_judge_copy_and_messages_to_text_agree_on_a_full_transcript():
    messages = build_assistant_message(_incomplete_response())
    judge_copy = drop_unreplayable_tool_records(messages, incomplete_as_text=True)

    def marked(text):
        return [line for line in text.splitlines() if 'no result recorded' in line]

    assert marked(messages_to_text(messages)) == ['[tool_call: refund({}) -> no result recorded]']
    assert marked(judge_copy[0].content) == marked(messages_to_text(messages))


def test_an_unflagged_call_without_a_tool_row_reads_incomplete_over_a_full_transcript():
    messages = [_user(), _assistant('c1')]

    assert 'no result recorded' in messages_to_text(messages)


def test_a_real_tool_row_wins_over_the_incomplete_flag():
    flagged = Message(role='assistant', tool_calls=[_call('c1').model_copy(update={'incomplete': True})])

    assert 'no result recorded' not in messages_to_text([flagged, _tool('c1')])
    assert 'no result recorded' in messages_to_text([{'role': 'assistant', 'tool_calls': [(flagged.tool_calls or [])[0]]}])


def test_duplicate_call_id_in_one_row_keeps_the_first_and_warns_once(caplog):
    messages = [
        _user(),
        Message(role='assistant', tool_calls=[_call('a', '{"x":1}'), _call('a', '{"x":2}')]),
        _tool('a'),
    ]

    with conversation_thread(), caplog.at_level(logging.WARNING):
        replay = drop_unreplayable_tool_records(messages)

    assert [tc.function.arguments for tc in replay[1].tool_calls or []] == ['{"x":1}']
    assert len([r for r in caplog.records if 'Not replaying' in r.message]) == 1


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
    response = AgentResponse(output=[ToolCallOutputItem(name='refund', arguments='{}', result='approved')])
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


@pytest.mark.parametrize('incomplete_as_text', [False, True])
def test_unknown_arguments_stay_unknown_on_the_copy_and_go_out_as_an_empty_object(incomplete_as_text):
    messages = [_user(), Message(role='assistant', tool_calls=[_call('c1', UNKNOWN_ARGUMENTS)]), _tool('c1')]
    snapshot = [m.model_copy(deep=True) for m in messages]

    replay = drop_unreplayable_tool_records(messages, incomplete_as_text=incomplete_as_text)
    items = messages_to_responses_input(replay)

    assert [tc.function.arguments for tc in replay[1].tool_calls or []] == [UNKNOWN_ARGUMENTS]
    assert replay[1].to_chat_completion()['tool_calls'][0]['function']['arguments'] == '{}'
    assert [i['arguments'] for i in items if i.get('type') == 'function_call'] == ['{}']
    assert messages == snapshot
    assert 'arguments unavailable' in messages_to_text(messages)
    assert ('arguments unavailable' in str(replay[1].content)) == incomplete_as_text


@pytest.mark.parametrize(
    ('given', 'sent'), [('', '{}'), ('{"a":1}', '{"a":1}'), ('{"a":', '{}'), ('"x"', '{}'), ('[1]', '{}')]
)
def test_serializers_send_an_object(given, sent):
    message = Message(role='assistant', tool_calls=[_call('c1', given)])

    assert message.to_chat_completion()['tool_calls'][0]['function']['arguments'] == sent
    assert messages_to_responses_input([message, _tool('c1')])[0]['arguments'] == sent
    assert messages_to_responses_input([message], wire=False)[0]['arguments'] == given


def test_no_serialized_payload_has_an_empty_assistant_message():
    messages = [_user(), Message(role='assistant', content=''), _user(), _assistant('gone'), _user()]

    replay = drop_unreplayable_tool_records(messages)
    chat = [m.to_chat_completion() for m in replay]
    items = messages_to_responses_input(replay)

    assert all(m['content'] for m in chat if m['role'] == 'assistant')
    assert all(i['content'][0]['text'] for i in items if i.get('role') == 'assistant')
