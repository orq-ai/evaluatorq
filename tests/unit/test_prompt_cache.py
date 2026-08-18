"""Prompt-cache breakpoint placement."""

from __future__ import annotations

import json

import pytest

from evaluatorq.common.prompt_cache import (
    CACHE_CONTROL_EPHEMERAL,
    apply_cache_breakpoints,
    mark_responses_input,
)
from evaluatorq.common.tracing import _serialize_messages  # pyright: ignore[reportPrivateUsage]


def _blocks(message: dict[str, object]) -> list[dict[str, object]]:
    content = message['content']
    assert isinstance(content, list)
    return content


def test_system_and_last_message_are_marked() -> None:
    marked = apply_cache_breakpoints(
        [
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'again'},
        ],
        volatile_tail=0,
    )

    assert _blocks(marked[0]) == [{'type': 'text', 'text': 'rules', 'cache_control': {'type': 'ephemeral'}}]
    assert _blocks(marked[3])[0]['text'] == 'again'
    # Middle turns stay plain strings: max 4 breakpoints, and the two ends cover
    # the static prefix and the growing transcript.
    assert marked[1]['content'] == 'hi'
    assert marked[2]['content'] == 'hello'


def test_unmarkable_shapes_are_left_alone() -> None:
    original = [
        {'role': 'system', 'content': ''},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x'}]},
    ]
    assert apply_cache_breakpoints(original, volatile_tail=0) == original

    # Caller-built blocks own their own breakpoints; don't clobber them.
    prebuilt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'a'}]}]
    assert apply_cache_breakpoints(prebuilt, volatile_tail=0) == prebuilt


def test_input_is_not_mutated() -> None:
    original = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]
    apply_cache_breakpoints(original, volatile_tail=0)
    assert original == [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]


def test_empty_list_is_a_no_op() -> None:
    """`len(out) - 1 == -1` used to leave index 0 in the candidate set."""
    assert apply_cache_breakpoints([], volatile_tail=0) == []


def test_cache_control_is_copied_not_shared() -> None:
    """One shared dict under every block is an aliasing trap; also `json.dumps`
    must accept it, which rules out a MappingProxyType."""
    marked_messages = apply_cache_breakpoints([{'role': 'user', 'content': 'x'}], volatile_tail=0)
    marked = _blocks(marked_messages[0])[0]['cache_control']
    assert marked == CACHE_CONTROL_EPHEMERAL and marked is not CACHE_CONTROL_EPHEMERAL
    json.dumps(marked_messages)


def test_single_message_is_marked_once() -> None:
    marked = apply_cache_breakpoints([{'role': 'user', 'content': 'hi'}], volatile_tail=0)
    assert len(marked) == 1
    assert _blocks(marked[0])[0]['text'] == 'hi'


def test_span_serialization_flattens_marked_content() -> None:
    """A block list must not reach the span as a Python repr — a judge or a human
    reading the trace would see cache_control noise instead of the prompt."""
    serialized = _serialize_messages(
        apply_cache_breakpoints([{'role': 'system', 'content': 'rules'}], volatile_tail=0)
    )
    assert '"content": "rules"' in serialized
    assert 'cache_control' not in serialized


def test_volatile_tail_keeps_the_breakpoint_off_a_per_turn_message() -> None:
    """The judge rebuilds its trailing instruction every turn. Marked, the next
    turn puts transcript content at that position and the prefix diverges right
    after the system message — a full write, read back never."""
    marked = apply_cache_breakpoints(
        [
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'persisted'},
            {'role': 'user', 'content': 'rebuilt every turn'},
        ],
        volatile_tail=1,
    )

    assert _blocks(marked[1])[0]['text'] == 'persisted'
    assert marked[2]['content'] == 'rebuilt every turn'


def test_breakpoint_walks_back_past_an_unmarkable_trailing_turn() -> None:
    """A transcript ending in an assistant reply (or a tool result) would
    otherwise get no prefix breakpoint at all, silently."""
    marked = apply_cache_breakpoints(
        [
            {'role': 'system', 'content': 'rules'},
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'answer'},
        ],
        volatile_tail=0,
    )

    assert _blocks(marked[1])[0]['text'] == 'question'
    assert marked[2]['content'] == 'answer'


def test_no_markable_prefix_leaves_everything_alone() -> None:
    original = [{'role': 'assistant', 'content': 'only turn'}]
    assert apply_cache_breakpoints(original, volatile_tail=0) == original


def test_volatile_tail_longer_than_the_prefix_is_a_no_op() -> None:
    original = [{'role': 'user', 'content': 'hi'}]
    assert apply_cache_breakpoints(original, volatile_tail=5) == original


def test_negative_volatile_tail_is_rejected() -> None:
    with pytest.raises(ValueError, match='volatile_tail'):
        apply_cache_breakpoints([{'role': 'user', 'content': 'hi'}], volatile_tail=-1)


def test_volatile_tail_has_no_default() -> None:
    """Required keyword on purpose: a caller that rebuilds its last message and
    forgets to say so gets a per-turn write and no read — a bill, not a crash."""
    with pytest.raises(TypeError, match='volatile_tail'):
        apply_cache_breakpoints([{'role': 'user', 'content': 'hi'}])  # pyright: ignore[reportCallIssue]


def test_responses_marks_the_end_of_the_prefix() -> None:
    items = [
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'persisted'}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'rebuilt'}]},
    ]
    marked = mark_responses_input(items, volatile_tail=1)

    assert marked[0]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
    assert 'cache_control' not in marked[1]['content'][-1]
    # Copied, not shared: one object under every marked part of every in-flight
    # request is an aliasing trap the first per-part TTL override would spring.
    assert marked[0]['content'][-1]['cache_control'] is not CACHE_CONTROL_EPHEMERAL
    assert 'cache_control' not in items[0]['content'][0], 'input must not be mutated'


def test_responses_promotes_bare_string_content_to_a_part() -> None:
    """`messages_to_responses_input` renders a plain user turn as a bare string;
    only an assistant turn arrives as a parts list."""
    marked = mark_responses_input([{'role': 'user', 'content': 'hello'}], volatile_tail=0)

    assert marked[0]['content'] == [
        {'type': 'input_text', 'text': 'hello', 'cache_control': {'type': 'ephemeral'}}
    ]


def test_responses_skips_function_call_items() -> None:
    """A `function_call` item has no `role` and no markable content; the
    breakpoint walks back to the message before it."""
    items = [
        {'role': 'user', 'content': 'ask'},
        {'type': 'function_call', 'call_id': 'c1', 'name': 'f', 'arguments': '{}'},
    ]
    marked = mark_responses_input(items, volatile_tail=0)

    assert marked[0]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
    assert marked[1] == items[1]


def test_responses_with_no_markable_item_is_a_no_op() -> None:
    items = [{'type': 'function_call', 'call_id': 'c1', 'name': 'f', 'arguments': '{}'}]
    assert mark_responses_input(items, volatile_tail=0) == items
    assert mark_responses_input([], volatile_tail=0) == []


def test_responses_negative_volatile_tail_is_rejected() -> None:
    with pytest.raises(ValueError, match='volatile_tail'):
        mark_responses_input([{'role': 'user', 'content': 'hi'}], volatile_tail=-1)


def test_responses_volatile_tail_has_no_default() -> None:
    with pytest.raises(TypeError, match='volatile_tail'):
        mark_responses_input([{'role': 'user', 'content': 'hi'}])  # pyright: ignore[reportCallIssue]
