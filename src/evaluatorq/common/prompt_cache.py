"""Anthropic prompt-cache breakpoints for multi-turn conversations.

Anthropic caching is **opt-in**: without a ``cache_control`` breakpoint there is
no caching at all, however byte-stable the prefix is. An append-only transcript
buys nothing on its own — it only makes a breakpoint worth placing.

OpenAI and Gemini 2.0+ cache the prefix automatically, so there is no
per-provider branch here and OpenAI's ``prompt_cache_key`` is not set anywhere
— it is a routing hint, not an enabler, and caching there needs no request
change.

**Callers must gate these on** ``llm_client.client_routes_through_orq``. The
marker is only known to be tolerated by the Orq router; ``cache_control`` inside
a content part is not in the direct OpenAI schema (the installed SDK's
``ChatCompletionContentPartTextParam`` defines ``type`` and ``text`` only), and
no one has checked what a self-hosted OpenAI-compatible server does with it.

Only apply these to a conversation that is replayed with a growing prefix (an
agent-simulation turn loop, a red-team attack thread). A cache **write** costs
1.25x the input tokens, so marking a one-shot prompt is a straight loss.

Placement rules (Orq docs, ``/docs/ai-gateway/features/prompt-caching``): the
breakpoint marks the *end* of the cacheable prefix, max 4 per request, and
caching only engages once the prefix clears the model's minimum (1-4k tokens
depending on the model) — below that it silently does nothing.

Both APIs take a **positioned, per-item** marker — `apply_cache_breakpoints` for
Chat Completions, `mark_responses_input` for Responses — so `volatile_tail` works
the same on either. Responses also accepts a *top-level* ``cache_control`` body
field, which marks the end of the whole input and therefore cannot be kept off a
rebuilt trailing item; it is deliberately not used here. Measured on
``anthropic/claude-sonnet-4-6`` with a uuid-salted cold prefix, three judgements
each (``scripts/manual_tests/prompt_cache_judge_check.py``):

===================  ======  ======  ======
path                 call 1  call 2  call 3
===================  ======  ======  ======
chat_completions          0   6,991   7,881
responses                 0   7,417   8,304
responses, top-level      0       0       0
===================  ======  ======  ======

Never set ``ttl``: 5m is the default, 1h costs more, and only Anthropic honours
it.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {'type': 'ephemeral'}
"""The only cache type Anthropic supports. Default TTL is 5 minutes.

Copied — never embedded by reference — into a request body: one shared object
under every marked block of every in-flight request is an aliasing trap the
first per-block TTL override would spring. It stays a plain ``dict`` rather
than a `MappingProxyType` because it is serialised by `json.dumps`, which
rejects a mapping proxy.
"""

# Only these carry a plain-text body we can safely re-render as a block list.
# `tool` and tool-calling `assistant` turns are left alone: their content shape
# is load-bearing for the provider and a breakpoint there buys nothing extra.
_MARKABLE_ROLES = frozenset({'system', 'user'})


def cached_text_block(text: str) -> dict[str, Any]:
    """Render ``text`` as a text content block carrying a cache breakpoint."""
    return {'type': 'text', 'text': text, 'cache_control': dict(CACHE_CONTROL_EPHEMERAL)}


def _is_markable(message: dict[str, Any]) -> bool:
    content = message.get('content')
    return message.get('role') in _MARKABLE_ROLES and isinstance(content, str) and bool(content)


def apply_cache_breakpoints(messages: list[dict[str, Any]], *, volatile_tail: int) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with breakpoints on the system + prefix end.

    Two of the four allowed breakpoints:

    - the leading ``system`` message — static for the whole conversation, so it
      is a cache read on every turn after the first;
    - the end of the **persisted** prefix, so the *next* turn (which appends past
      it) reads the whole transcript back.

    ``volatile_tail`` is the number of trailing messages the caller rebuilds on
    every turn rather than appending to a transcript — a per-call instruction, a
    re-rendered scratchpad. They are excluded from the prefix. Marking one is
    worse than marking nothing: the next turn puts persisted content at that
    position, the prefix diverges immediately after the system message, and the
    whole transcript pays a 1.25x write it can never read back.

    It is **required, with no default**, on purpose: a caller that rebuilds its
    last message and does not say so gets a silent per-turn write and no read,
    which is the expensive failure and shows up as a bill rather than a bug. Pass
    ``0`` when the whole list persists into the next turn.

    A message is skipped when its role is not ``system``/``user`` or its content
    is not a non-empty string — a caller that already built content blocks owns
    its own breakpoints, and re-wrapping would clobber them. The prefix
    breakpoint walks backwards past unmarkable trailing turns (an ``assistant``
    reply, a ``tool`` result) to the nearest one that can carry it.

    Raises:
        ValueError: if ``volatile_tail`` is negative.
    """
    if volatile_tail < 0:
        raise ValueError(f'volatile_tail must be >= 0, got {volatile_tail}')
    out = list(messages)
    prefix_end = len(out) - 1 - volatile_tail
    while prefix_end >= 0 and not _is_markable(out[prefix_end]):
        prefix_end -= 1
    if prefix_end < 0 and out:
        logger.debug(
            'No cacheable message in the prefix ({} messages, volatile_tail={}) — only the system '
            'breakpoint (if any) is placed and the transcript is re-encoded every turn.',
            len(out),
            volatile_tail,
        )
    # The leading breakpoint is the *system* prompt specifically — a leading user
    # message is already covered by the prefix breakpoint, and marking it when the
    # whole conversation is inside `volatile_tail` would mark a message the caller
    # just told us is rebuilt every turn. Set, not a pair: dedupes when the prefix
    # end *is* the system message, and empties out on an empty list.
    leading = 0 if out and out[0].get('role') == 'system' else -1
    for index in {i for i in (leading, prefix_end) if 0 <= i < len(out) and _is_markable(out[i])}:
        out[index] = {**out[index], 'content': [cached_text_block(out[index]['content'])]}
    return out


def mark_responses_input(input_items: list[dict[str, Any]], *, volatile_tail: int) -> list[dict[str, Any]]:
    """Return a copy of a Responses ``input`` list with a positioned breakpoint.

    The Responses `input` is a list of items carrying content parts, exactly like
    a chat message list, and the router honours a **per-item** ``cache_control``
    on the last part of an item. That is what makes `volatile_tail` work here too
    — measured against ``anthropic/claude-sonnet-4-6``: marking the end of the
    persisted prefix reads 4,764 tokens back on the next call, while the
    top-level switch alone reads 0 because it marks the end of the whole input,
    including a trailing item the caller rebuilt.

    Preferred over the top-level ``cache_control`` body field, which marks the end
    of the whole input: the two compose, but the top-level one then adds a write
    at the end that nothing reads back.
    """
    if volatile_tail < 0:
        raise ValueError(f'volatile_tail must be >= 0, got {volatile_tail}')
    out = list(input_items)
    index = len(out) - 1 - volatile_tail
    while index >= 0 and not _has_markable_parts(out[index]):
        index -= 1
    if index < 0:
        if out:
            logger.debug(
                'No cacheable item in the Responses prefix ({} items, volatile_tail={}) — the input '
                'is re-encoded every turn.',
                len(out),
                volatile_tail,
            )
        return out
    content = out[index]['content']
    # `messages_to_responses_input` renders a plain user turn as a bare string and
    # only an assistant turn as parts, so both shapes arrive here. A string is
    # promoted to a single `input_text` part, which the API accepts for every
    # non-assistant role (an assistant turn is always already a part list).
    parts: list[dict[str, Any]] = (
        [{'type': 'input_text', 'text': content}] if isinstance(content, str) else [dict(part) for part in content]
    )
    parts[-1]['cache_control'] = dict(CACHE_CONTROL_EPHEMERAL)
    out[index] = {**out[index], 'content': parts}
    return out


def _has_markable_parts(item: dict[str, Any]) -> bool:
    """A message item carrying text we can attach a breakpoint to.

    Excludes `function_call` / `function_call_output` items, which have no
    ``content`` and whose shape is load-bearing for the API.
    """
    content = item.get('content')
    if 'role' not in item:
        return False
    if isinstance(content, str):
        return bool(content)
    return isinstance(content, list) and bool(content) and all(isinstance(part, dict) for part in content)
