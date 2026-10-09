from __future__ import annotations

import re

import pytest

from evaluatorq.common.model_input import (
    JEV_STATE_ALL_QUESTIONS_CHARS,
    JEV_STATE_CHARS,
    JEV_STATE_QUESTION_CHARS,
    cap_text,
    classifier_question_wire_payloads,
    fit_jev_state,
    is_jev_model,
    jev_state,
    serialized_chars,
    serialized_question_chars,
)


def test_global_text_cap_keeps_unicode_edges_and_exact_marker() -> None:
    source = 'α' * 100

    bounded, omitted = cap_text(source, 40)
    assert bounded == f'{"α" * 6}[... 87 chars left out ...]{"α" * 7}'
    assert len(bounded) == 40
    assert omitted == 87
    assert source == 'α' * 100


def test_literal_omission_marker_is_not_counted_as_prior_truncation() -> None:
    literal = '[... 100000 chars left out ...]'
    source = literal + 'x' * 1_000

    bounded, omitted = cap_text(source, 100)

    assert literal in bounded
    assert omitted < len(source)


def test_question_wire_payloads_count_exact_fields_and_are_idempotent() -> None:
    questions = {
        'boolean': {
            'kind': 'noul',
            'instructions': 'Is this correct?',
            'criteria': None,
            'state': {'not': 'serialized'},
            'noul_threshold': 0.5,
        },
        'choice': {
            'type': 'choice',
            'instructions': 'Choose one.',
            'criteria': {'a': 'First', 'b': 'Second'},
            'state': {'also': 'not serialized'},
        },
    }
    expected = {
        'boolean': {'type': 'noul', 'instructions': 'Is this correct?'},
        'choice': {'type': 'choice', 'instructions': 'Choose one.', 'criteria': {'a': 'First', 'b': 'Second'}},
    }

    assert classifier_question_wire_payloads(questions) == expected
    assert classifier_question_wire_payloads(expected) == expected
    assert serialized_question_chars(questions) == serialized_chars(expected)

def test_jev_state_keeps_user_edges_and_does_not_mutate_source() -> None:
    source = [{'role': 'user', 'content': 'x' * 3_000}]

    state = jev_state(source, global_char_cap=500_000)

    assert state['messages'][0]['text'] == f'{"x" * 1000}[... 1000 chars left out ...]{"x" * 1000}'
    assert source[0]['content'] == 'x' * 3_000


def test_jev_tool_call_pairs_every_matching_result_and_retains_status() -> None:
    source = [
        {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'function': {'name': 'shell.exec', 'arguments': 'x' * 300}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'b' * 300, 'status': 'failed', 'exit_code': 7},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'second result', 'status': 'completed'},
        {'role': 'tool', 'content': 'orphan'},
    ]

    state = jev_state(source, global_char_cap=500_000)
    calls = [item for item in state['messages'] if item['type'] == 'tool_call']

    assert calls[0]['name'] == 'shell.exec'
    assert calls[0]['input'] == f'{"x" * 100}[... 100 chars left out ...]{"x" * 100}'
    assert len(calls[0]['results']) == 2
    assert calls[0]['results'][0]['status'] == 'failed'
    assert calls[0]['results'][0]['exit_code'] == 7
    assert any(item['type'] == 'orphan_result' for item in state['messages'])


def test_jev_duplicate_tool_call_ids_pair_results_in_source_order() -> None:
    source = [
        {
            'role': 'assistant',
            'tool_calls': [
                {'id': 'call-duplicate', 'function': {'name': 'first', 'arguments': 'a'}},
                {'id': 'call-duplicate', 'function': {'name': 'second', 'arguments': 'b'}},
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call-duplicate', 'content': 'first result'},
        {'role': 'tool', 'tool_call_id': 'call-duplicate', 'content': 'second result'},
    ]

    state = jev_state(source, global_char_cap=500_000)
    calls = [item for item in state['messages'] if item['type'] == 'tool_call']

    assert [call['name'] for call in calls] == ['first', 'second']
    assert [[result['text'] for result in call['results']] for call in calls] == [
        ['first result'],
        ['second result'],
    ]


def test_jev_reused_tool_call_id_uses_latest_assistant_group() -> None:
    source = [
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'call-reused', 'function': {'name': 'earlier', 'arguments': 'a'}}],
        },
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'call-reused', 'function': {'name': 'later', 'arguments': 'b'}}],
        },
        {'role': 'tool', 'tool_call_id': 'call-reused', 'content': 'latest result'},
    ]

    state = jev_state(source, global_char_cap=500_000)
    calls = [item for item in state['messages'] if item['type'] == 'tool_call']

    assert [call['name'] for call in calls] == ['earlier', 'later']
    assert calls[0]['results'] == []
    assert [result['text'] for result in calls[1]['results']] == ['latest result']



def test_jev_model_gate_and_question_capacity_validation() -> None:
    assert is_jev_model('typesafe/jev-latest')
    assert not is_jev_model('openai/gpt-6-luna')

    with pytest.raises(ValueError, match='questions alone'):
        fit_jev_state({'messages': []}, [{'instructions': 'q' * JEV_STATE_ALL_QUESTIONS_CHARS}])


def test_jev_global_cap_keeps_complete_head_and_tail_messages() -> None:
    source = [{'role': 'user', 'content': f'message-{index}-' + 'x' * 100} for index in range(20)]

    state = jev_state(source, global_char_cap=1_000)

    assert state['messages'][0]['index'] == 0
    assert state['messages'][-1]['index'] == 19
    assert state['omission'] == f'[... {20 - len(state["messages"])} messages left out ...]'


def test_long_question_reserves_state_capacity() -> None:
    state = {
        'messages': [
            {'index': index, 'role': 'user', 'type': 'message', 'text': 'x' * 1_000} for index in range(200)
        ]
    }

    bounded = fit_jev_state(state, [{'instructions': 'q' * 100_000}])

    assert serialized_chars(bounded) <= 28_000 * 4 - 100_000


@pytest.mark.parametrize(
    ('role', 'excerpt_chars', 'edge_chars'),
    [('system', 8_000, 4_000), ('developer', 8_000, 4_000), ('user', 2_000, 1_000), ('assistant', 1_000, 500)],
)
def test_jev_role_excerpts_keep_exact_edges_without_mutating_source(
    role: str, excerpt_chars: int, edge_chars: int
) -> None:
    original = 'begin-' + 'x' * (excerpt_chars + 100) + '-end'
    source = [{'role': role, 'content': original}]

    state = jev_state(source, global_char_cap=500_000)

    rendered = state['messages'][0]['text']
    omitted = len(original) - edge_chars * 2
    assert rendered == f'{original[:edge_chars]}[... {omitted} chars left out ...]{original[-edge_chars:]}'
    assert source[0]['content'] == original


def test_jev_global_cap_keeps_edges_of_one_oversized_user_message_before_role_excerpt() -> None:
    original = 'user-head:' + 'x' * 600_000 + ':user-tail'
    source = [{'role': 'user', 'content': original}]

    state = jev_state(source, global_char_cap=500_000)
    state = fit_jev_state(state, [])

    rendered = state['messages'][0]['text']
    assert rendered.startswith('user-head:')
    assert rendered.endswith(':user-tail')
    assert '[... ' in rendered
    assert rendered == f'{original[:1000]}[... {len(original) - 2_000} chars left out ...]{original[-1000:]}'
    assert serialized_chars(state) <= 500_000
    assert source[0]['content'] == original


def test_jev_global_cap_keeps_oversized_tool_output_edges_and_result_metadata() -> None:
    output = 'tool-head:' + 'y' * 600_000 + ':tool-tail'
    arguments = 'arg-head-' + 'x' * 600_000 + '-arg-tail'
    source = [
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'shell-1', 'function': {'name': 'shell.exec', 'arguments': arguments}}],
        },
        {
            'role': 'tool',
            'tool_call_id': 'shell-1',
            'content': output,
            'trace_finder_metadata': {'tool_result': {'status': 'failed', 'error_code': 'EPIPE'}},
            'is_error': True,
            'exit_code': 23,
        },
    ]

    state = fit_jev_state(jev_state(source, global_char_cap=500_000), [])

    call = next(entry for entry in state['messages'] if entry['type'] == 'tool_call')
    assert call['name'] == 'shell.exec'
    assert call['input'] == f'{arguments[:100]}[... {len(arguments) - 200} chars left out ...]{arguments[-100:]}'
    result = call['results'][0]
    assert result['text'].startswith('tool-head:')
    assert result['text'].endswith(':tool-tail')
    assert result['text'] == f'{output[:100]}[... {len(output) - 200} chars left out ...]{output[-100:]}'
    assert result['status'] == 'failed'
    assert result['error_code'] == 'EPIPE'
    assert result['exit_code'] == 23
    assert serialized_chars(state) <= 500_000
    assert source[1]['content'] == output


def test_jev_omission_marker_counts_unrepresented_source_messages_and_fits_global_cap() -> None:
    source = [{'role': 'user', 'content': f'message-{index}-' + 'x' * 100} for index in range(20)]

    state = jev_state(source, global_char_cap=1_000)

    represented = {entry['index'] for entry in state['messages']}
    assert state['omission'] == f'[... {len(source) - len(represented)} messages left out ...]'
    assert 0 in represented and len(source) - 1 in represented
    assert serialized_chars(state) <= 1_000


def test_jev_final_omission_marker_stays_within_question_limited_state_budget() -> None:
    source = [{'role': 'user', 'content': f'message-{index}-' + 'x' * 2_000} for index in range(80)]
    questions = [{'instructions': 'q' * 30_000}]

    state = fit_jev_state(jev_state(source, global_char_cap=500_000), questions)

    question_chars = serialized_chars(questions[0])
    assert serialized_chars(state) <= 28_000 * 4 - question_chars
    represented = {entry['index'] for entry in state['messages']}
    omitted = len(source) - len(represented)
    assert state['omission'] == f'[... {omitted} messages left out ...]'


def test_jev_removes_harness_reminders_and_typed_reasoning_but_keeps_business_json() -> None:
    source = [
        {
            'role': 'assistant',
            'content': [
                {'type': 'reasoning', 'text': 'private chain of thought'},
                {'type': 'text', 'text': 'answer <system-reminder>ignore this</system-reminder> safely'},
            ],
        },
        {'role': 'user', 'content': {'reasoning': 'application data', 'question': 'help'}},
    ]

    state = jev_state(source, global_char_cap=500_000)
    rendered = str(state)

    assert 'private chain of thought' not in rendered
    assert 'ignore this' not in rendered
    assert 'application data' in rendered
    assert 'help' in rendered
    assert 'harness reminder omitted' in rendered

def test_structured_text_data_and_url_fields_are_not_mislabeled_as_media() -> None:
    content = {'text': 'label', 'data': {'records': [{'id': 7}]}, 'url': '/records/7'}

    state = jev_state([{'role': 'user', 'content': content}], global_char_cap=500_000)

    message = state['messages'][0]
    assert 'media' not in message
    assert '"label"' in message['text']
    assert '"records"' in message['text']
    assert '"/records/7"' in message['text']


def test_literal_omission_marker_does_not_inflate_jev_excerpt_count() -> None:
    literal = '[... 100000 chars left out ...]'
    source = literal + 'x' * 600_000

    state = fit_jev_state(jev_state([{'role': 'user', 'content': source}], global_char_cap=500_000), [])

    text = state['messages'][0]['text']
    markers = re.findall(r'\[\.\.\. (\d+) chars left out \.\.\.\]', text)
    assert markers
    assert int(markers[-1]) == len(source) - 2_000
    assert text.startswith(literal)



def test_jev_inline_base64_media_reports_decoded_size_without_copying_payload() -> None:
    import base64

    payload = b'\x89PNG' + b'x' * 130_000
    encoded = base64.b64encode(payload).decode('ascii')
    source = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}}]}]

    state = jev_state(source, global_char_cap=500_000)

    assert state['messages'][0]['media'] == [f'[image, {len(payload) // 1024} KB]']
    assert encoded not in str(state)


def test_jev_message_and_tool_results_preserve_multimodal_order() -> None:
    content = [
        {'type': 'text', 'text': 'refer to'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.test/image.png'}},
        {'type': 'text', 'text': 'above'},
    ]
    source = [
        {'role': 'user', 'content': content},
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'call-ordered', 'function': {'name': 'read', 'arguments': '{}'}}],
        },
        {'role': 'tool', 'tool_call_id': 'call-ordered', 'content': content},
    ]

    state = jev_state(source, global_char_cap=500_000)
    message = next(entry for entry in state['messages'] if entry['role'] == 'user')
    call = next(entry for entry in state['messages'] if entry['type'] == 'tool_call')
    expected = 'refer to\n[image, size unknown]\nabove'

    assert message['text'] == expected
    assert message['media'] == ['[image, size unknown]']
    assert call['results'][0]['text'] == expected



def test_question_limited_jev_state_preserves_and_counts_top_level_metadata() -> None:
    state = {
        'trace_status': 'completed',
        'messages': [{'index': index, 'role': 'user', 'type': 'message', 'text': 'x' * 2_000} for index in range(80)],
    }
    questions = [{'instructions': 'q' * 30_000}]

    bounded = fit_jev_state(state, questions)

    assert bounded['trace_status'] == 'completed'
    question_chars = serialized_chars(questions)
    longest_question = max(serialized_chars(question) for question in questions)
    assert serialized_chars(bounded) <= JEV_STATE_CHARS
    assert serialized_chars(bounded) + longest_question <= JEV_STATE_QUESTION_CHARS
    assert serialized_chars(bounded) + question_chars <= JEV_STATE_ALL_QUESTIONS_CHARS
    assert len(bounded['messages']) < len(state['messages'])



def test_question_fit_keeps_the_full_state_ceiling_and_respects_question_limits() -> None:
    source = [{'role': 'user', 'content': f'event-{index}-' + 'x' * 2_000} for index in range(50)]
    state = jev_state(source, global_char_cap=500_000)
    questions = [{'instructions': 'q' * 16_000}]

    bounded = fit_jev_state(state, questions)

    question_chars = serialized_chars(questions)
    longest_question = max(serialized_chars(question) for question in questions)
    assert serialized_chars(state) > JEV_STATE_CHARS - longest_question
    assert bounded != state
    assert serialized_chars(bounded) <= JEV_STATE_CHARS - question_chars
    assert serialized_chars(bounded) + longest_question <= JEV_STATE_QUESTION_CHARS
    assert serialized_chars(bounded) + question_chars <= JEV_STATE_ALL_QUESTIONS_CHARS
def test_jev_role_excerpt_accumulates_global_and_role_omissions() -> None:
    original = 'x' * 600_000
    state = fit_jev_state(jev_state([{'role': 'user', 'content': original}], global_char_cap=500_000), [])

    assert state['messages'][0]['text'] == f'{original[:1000]}[... {len(original) - 2_000} chars left out ...]{original[-1000:]}'
