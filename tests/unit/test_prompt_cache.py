"""Prompt-cache breakpoint placement."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from evaluatorq.common.prompt_cache import (
    CACHE_MIN_PROMPT_TOKENS,
    apply_cache_breakpoints,
    caching_applies,
    mark_responses_input,
    responses_volatile_items,
)
from evaluatorq.common.tracing import _serialize_messages  # pyright: ignore[reportPrivateUsage]
from evaluatorq.simulation.types import Message

# Every fixture must clear the size guard, or the helpers correctly place nothing
# and an assertion about placement would pass for the wrong reason.
_PAD = 'x ' * (CACHE_MIN_PROMPT_TOKENS * 2)


def _long(text: str) -> str:
    return f'{text} {_PAD}'


def _blocks(message: dict[str, object]) -> list[dict[str, object]]:
    content = message['content']
    assert isinstance(content, list)
    return content


def _client(base_url: str = 'https://my.orq.ai/v3/router') -> MagicMock:
    client = MagicMock()
    client.base_url = base_url
    return client


def test_system_and_last_message_are_marked() -> None:
    marked = apply_cache_breakpoints(
        [
            {'role': 'system', 'content': _long('rules')},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'again'},
        ],
        volatile_tail=0,
    )

    assert _blocks(marked[0]) == [
        {'type': 'text', 'text': _long('rules'), 'cache_control': {'type': 'ephemeral'}}
    ]
    assert _blocks(marked[3])[0]['text'] == 'again'
    # Middle turns stay plain strings: max 4 breakpoints, and the two ends cover
    # the static prefix and the growing transcript.
    assert marked[1]['content'] == 'hi'
    assert marked[2]['content'] == 'hello'


def test_unmarkable_shapes_are_left_alone() -> None:
    original = [
        {'role': 'system', 'content': ''},
        {'role': 'assistant', 'content': _long('reply'), 'tool_calls': [{'id': 'x'}]},
    ]
    assert apply_cache_breakpoints(original, volatile_tail=0) == original

    # Caller-built blocks own their own breakpoints; don't clobber them.
    prebuilt = [{'role': 'user', 'content': [{'type': 'text', 'text': _long('a')}]}]
    assert apply_cache_breakpoints(prebuilt, volatile_tail=0) == prebuilt


def test_input_is_not_mutated() -> None:
    original = [{'role': 'system', 'content': _long('rules')}, {'role': 'user', 'content': 'hi'}]
    snapshot = [dict(m) for m in original]
    apply_cache_breakpoints(original, volatile_tail=0)
    assert original == snapshot


def test_empty_list_is_a_no_op() -> None:
    """`len(out) - 1 == -1` used to leave index 0 in the candidate set."""
    assert apply_cache_breakpoints([], volatile_tail=0) == []


def test_a_prompt_below_the_minimum_is_not_marked() -> None:
    """A write costs 1.25x and no Anthropic model caches below the floor, so a
    breakpoint on a short exchange is a guaranteed loss."""
    original = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]
    assert apply_cache_breakpoints(original, volatile_tail=0) == original


def test_each_marked_block_gets_its_own_cache_control() -> None:
    """One shared dict under every block is an aliasing trap the first per-block
    TTL override would spring; the payload must also survive `json.dumps`."""
    marked = apply_cache_breakpoints(
        [{'role': 'system', 'content': _long('rules')}, {'role': 'user', 'content': _long('hi')}],
        volatile_tail=0,
    )

    first = _blocks(marked[0])[0]['cache_control']
    second = _blocks(marked[1])[0]['cache_control']
    assert first == second == {'type': 'ephemeral'}
    assert first is not second
    json.dumps(marked)


def test_single_message_is_marked_once() -> None:
    marked = apply_cache_breakpoints([{'role': 'user', 'content': _long('hi')}], volatile_tail=0)
    assert len(marked) == 1
    assert _blocks(marked[0])[0]['text'] == _long('hi')


def test_span_serialization_flattens_marked_content() -> None:
    """A block list must not reach the span as a Python repr — a judge or a human
    reading the trace would see cache_control noise instead of the prompt."""
    serialized = _serialize_messages(
        apply_cache_breakpoints([{'role': 'system', 'content': _long('rules')}], volatile_tail=0)
    )
    assert f'"content": "{_long("rules")}"' in serialized
    assert 'cache_control' not in serialized


def test_volatile_tail_keeps_the_breakpoint_off_a_per_turn_message() -> None:
    """The judge rebuilds its trailing instruction every turn. Marked, the next
    turn puts transcript content at that position and the prefix diverges right
    after the system message — a full write, read back never."""
    marked = apply_cache_breakpoints(
        [
            {'role': 'system', 'content': _long('rules')},
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
            {'role': 'system', 'content': _long('rules')},
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'answer'},
        ],
        volatile_tail=0,
    )

    assert _blocks(marked[1])[0]['text'] == 'question'
    assert marked[2]['content'] == 'answer'


def test_no_markable_prefix_leaves_everything_alone() -> None:
    original = [{'role': 'assistant', 'content': _long('only turn')}]
    assert apply_cache_breakpoints(original, volatile_tail=0) == original


def test_volatile_tail_longer_than_the_prefix_still_marks_the_system_message() -> None:
    """Deliberate: the system message is built by the framework from a stable
    prompt, not rebuilt by the caller, so it is a valid read even when every
    caller-supplied message is volatile."""
    marked = apply_cache_breakpoints(
        [{'role': 'system', 'content': _long('rules')}, {'role': 'user', 'content': 'rebuilt'}],
        volatile_tail=2,
    )

    assert _blocks(marked[0])[0]['text'] == _long('rules')
    assert marked[1]['content'] == 'rebuilt'


def test_volatile_tail_longer_than_a_system_less_prefix_is_a_no_op() -> None:
    original = [{'role': 'user', 'content': _long('hi')}]
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
        {'role': 'user', 'content': [{'type': 'input_text', 'text': _long('persisted')}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': 'rebuilt'}]},
    ]
    marked = mark_responses_input(items, volatile_items=1)

    assert marked[0]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
    assert 'cache_control' not in marked[1]['content'][-1]
    assert 'cache_control' not in items[0]['content'][0], 'input must not be mutated'


def test_responses_promotes_bare_string_content_to_a_part() -> None:
    """`messages_to_responses_input` renders a plain user turn as a bare string;
    only an assistant turn arrives as a parts list."""
    marked = mark_responses_input([{'role': 'user', 'content': _long('hello')}], volatile_items=0)

    assert marked[0]['content'] == [
        {'type': 'input_text', 'text': _long('hello'), 'cache_control': {'type': 'ephemeral'}}
    ]


def test_responses_skips_function_call_items() -> None:
    """A `function_call` item has no `role` and no markable content; the
    breakpoint walks back to the message before it."""
    items = [
        {'role': 'user', 'content': _long('ask')},
        {'type': 'function_call', 'call_id': 'c1', 'name': 'f', 'arguments': '{}'},
    ]
    marked = mark_responses_input(items, volatile_items=0)

    assert marked[0]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
    assert marked[1] == items[1]


def test_responses_below_the_minimum_is_a_no_op() -> None:
    items = [{'role': 'user', 'content': 'hi'}]
    assert mark_responses_input(items, volatile_items=0) == items


def test_responses_with_no_markable_item_is_a_no_op() -> None:
    items = [{'type': 'function_call', 'call_id': 'c1', 'name': 'f', 'arguments': '{}'}]
    assert mark_responses_input(items, volatile_items=0) == items
    assert mark_responses_input([], volatile_items=0) == []


def test_responses_negative_volatile_items_is_rejected() -> None:
    with pytest.raises(ValueError, match='volatile_tail'):
        mark_responses_input([{'role': 'user', 'content': 'hi'}], volatile_items=-1)


def test_responses_volatile_items_has_no_default() -> None:
    with pytest.raises(TypeError, match='volatile_items'):
        mark_responses_input([{'role': 'user', 'content': 'hi'}])  # pyright: ignore[reportCallIssue]


def test_responses_volatile_items_counts_rendered_items_not_messages() -> None:
    """A tool-calling assistant turn renders to a content item plus one
    `function_call` per call, so a message count would under-count the tail."""
    from evaluatorq.contracts import FunctionCall, StrategyToolCall

    messages = [
        Message(role='user', content='persisted'),
        Message(
            role='assistant',
            content='calling',
            tool_calls=[StrategyToolCall(id='c1', function=FunctionCall(name='f', arguments='{}'))],
        ),
        Message(role='user', content='rebuilt'),
    ]

    assert responses_volatile_items(messages, volatile_tail=0) == 0
    assert responses_volatile_items(messages, volatile_tail=1) == 1
    assert responses_volatile_items(messages, volatile_tail=2) == 3


def test_responses_volatile_items_rejects_a_negative_tail() -> None:
    with pytest.raises(ValueError, match='volatile_tail'):
        responses_volatile_items([], volatile_tail=-1)


@pytest.mark.parametrize(
    ('base_url', 'model', 'expected'),
    [
        ('https://my.orq.ai/v3/router', 'anthropic/claude-sonnet-4-6', True),
        ('https://my.orq.ai/v3/router', 'claude-sonnet-4-6', True),
        # An Orq agent resolves its model server-side; we cannot see whether it is
        # Anthropic, and excluding it would leave the default target uncached.
        ('https://my.orq.ai/v3/router', 'agent/support-bot', True),
        ('https://my.orq.ai/v3/router', 'openai/gpt-4o', False),
        ('https://my.orq.ai/v3/router', 'google/gemini-2.5-pro', False),
        # Off the router the marker is outside the documented request schema.
        ('https://api.openai.com/v1', 'anthropic/claude-sonnet-4-6', False),
    ],
)
def test_caching_applies_needs_both_the_router_and_a_model_that_benefits(
    base_url: str, model: str, expected: bool
) -> None:
    assert caching_applies(_client(base_url), model) is expected


def test_caching_applies_is_false_without_a_client() -> None:
    assert caching_applies(None, 'anthropic/claude-sonnet-4-6') is False
