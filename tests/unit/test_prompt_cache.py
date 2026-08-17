"""Prompt-cache breakpoint placement."""

from __future__ import annotations

from evaluatorq.common.prompt_cache import apply_cache_breakpoints, responses_cache_body
from evaluatorq.common.tracing import _serialize_messages  # pyright: ignore[reportPrivateUsage]


def _blocks(message: dict[str, object]) -> list[dict[str, object]]:
    content = message['content']
    assert isinstance(content, list)
    return content


def test_system_and_last_message_are_marked() -> None:
    marked = apply_cache_breakpoints([
        {'role': 'system', 'content': 'rules'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'hello'},
        {'role': 'user', 'content': 'again'},
    ])

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
    assert apply_cache_breakpoints(original) == original

    # Caller-built blocks own their own breakpoints; don't clobber them.
    prebuilt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'a'}]}]
    assert apply_cache_breakpoints(prebuilt) == prebuilt


def test_input_is_not_mutated() -> None:
    original = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]
    apply_cache_breakpoints(original)
    assert original == [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'hi'}]


def test_single_message_is_marked_once() -> None:
    marked = apply_cache_breakpoints([{'role': 'user', 'content': 'hi'}])
    assert len(marked) == 1
    assert _blocks(marked[0])[0]['text'] == 'hi'


def test_span_serialization_flattens_marked_content() -> None:
    """A block list must not reach the span as a Python repr — a judge or a human
    reading the trace would see cache_control noise instead of the prompt."""
    serialized = _serialize_messages(apply_cache_breakpoints([{'role': 'system', 'content': 'rules'}]))
    assert '"content": "rules"' in serialized
    assert 'cache_control' not in serialized


def test_responses_cache_body_is_the_top_level_switch() -> None:
    assert responses_cache_body() == {'cache_control': {'type': 'ephemeral'}}
