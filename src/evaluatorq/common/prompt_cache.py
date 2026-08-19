"""Anthropic prompt-cache breakpoints for multi-turn conversations.

Anthropic caching is **opt-in**: without a ``cache_control`` breakpoint there is
no caching at all, however byte-stable the prefix is. An append-only transcript
buys nothing on its own — it only makes a breakpoint worth placing.

OpenAI, Gemini 2.0+, DeepSeek and xAI cache the prefix automatically, so they
need no request change and OpenAI's ``prompt_cache_key`` is not set anywhere —
it is a routing hint, not an enabler.

**Callers gate on** `caching_applies`, which requires *both* the Orq router and
an Anthropic model. The marker is only known to be tolerated by the router on an
Anthropic backend; ``cache_control`` inside a content part is not in the direct
OpenAI schema (the installed SDK's ``ChatCompletionContentPartTextParam`` defines
``type`` and ``text`` only), and no one has measured what a routed non-Anthropic
model or a self-hosted OpenAI-compatible server does with it.

Only apply these to a conversation that is replayed with a growing prefix (an
agent-simulation turn loop, a red-team attack thread). A cache **write** costs
1.25x the input tokens, so marking a one-shot prompt is a straight loss — which
is why both helpers refuse to mark an input below `CACHE_MIN_PROMPT_TOKENS`.

Placement rules (Orq docs, ``/docs/ai-gateway/features/prompt-caching``): the
breakpoint marks the *end* of the cacheable prefix and there are at most 4 per
request.

Both APIs take a **positioned, per-item** marker — `apply_cache_breakpoints` for
Chat Completions, `mark_responses_input` for Responses. Responses also accepts a
*top-level* ``cache_control`` body field, which marks the end of the whole input
and therefore cannot be kept off a rebuilt trailing item; it is deliberately not
used here. Measured on ``anthropic/claude-sonnet-4-6`` with a uuid-salted cold
prefix, three judgements each
(``scripts/manual_tests/prompt_cache_judge_check.py``):

| path                  | call 1 | call 2 | call 3 |
| --------------------- | ------ | ------ | ------ |
| chat_completions      |      0 |  6,991 |  7,881 |
| responses             |      0 |  7,417 |  8,304 |
| responses, top-level  |      0 |      0 |      0 |

Never set ``ttl``: 5m is the default, 1h costs more, and only Anthropic honours
it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.llm_client import client_routes_through_orq

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import AsyncOpenAI

CACHE_MIN_PROMPT_TOKENS = 1024
"""Below this, no Anthropic model caches and a breakpoint is a pure 1.25x loss.

A **necessary, not sufficient** condition: Orq's table gives per-model floors of
512 (Opus 5, Fable 5), 1024 (Sonnet 4.6/5, Opus 4.8), 2048 (Opus 4.7, Haiku 3.5)
and 4096 (Haiku 4.5, Opus 4.5/4.6). Gating on the lowest common floor that
applies to the models we actually run avoids a per-model table that would rot
with every release; a prompt between 1024 and its model's real floor still pays a
write nothing reads, which is the same behaviour as before this guard and strictly
better than paying it on a two-line prompt.
"""

_CHARS_PER_TOKEN = 4
"""Crude estimator, deliberately not a tokenizer.

The check only has to separate "a two-line one-shot prompt" from "a replayed
transcript", and importing a tokenizer for that would add a dependency and a
per-call cost to save nothing. It over-estimates tokens for text with long
words, which errs toward marking — the same behaviour as before this guard
existed.
"""

# Only these carry a plain-text body we can safely re-render as a block list.
# `tool` and tool-calling `assistant` turns are left alone: their content shape
# is load-bearing for the provider and a breakpoint there buys nothing extra.
_MARKABLE_ROLES = frozenset({'system', 'user'})


def caching_applies(client: AsyncOpenAI | None, model: str) -> bool:
    """True when a breakpoint on this ``client``/``model`` can pay off.

    The **router** is required: ``cache_control`` inside a content part is outside
    the direct OpenAI schema, and a self-hosted OpenAI-compatible server may
    reject the whole request rather than ignore the key. Orq's own docs are
    explicit that on the router it is safe either way — *"``cache_control`` on a
    non-Anthropic model is ignored, not rejected"*
    (``/docs/ai-gateway/features/anthropic-messages-api``) — so this second check
    is not a safety gate but a scope one: every other provider caches
    automatically, so marking them changes the request shape for no gain.

    ``agent/<key>`` is included because an Orq agent deployment resolves its model
    server-side: we cannot see whether it is Anthropic, and excluding it would
    leave the default red-team and simulation target — the largest replayed
    transcript in either surface — uncached. The documented ignore-don't-reject
    behaviour is what makes that safe.
    """
    if not client_routes_through_orq(client):
        return False
    name = model.lower()
    return name.startswith(('anthropic/', 'agent/')) or 'claude' in name


def _text_length(value: Any) -> int:
    """Total length of the text a message/item content carries, any shape."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(len(part.get('text', '')) for part in value if isinstance(part, dict))
    return 0


def _clears_minimum(items: list[dict[str, Any]]) -> bool:
    total = sum(_text_length(item.get('content')) for item in items)
    return total >= CACHE_MIN_PROMPT_TOKENS * _CHARS_PER_TOKEN


def _prefix_index(
    items: list[dict[str, Any]],
    volatile_tail: int,
    is_markable: Callable[[dict[str, Any]], bool],
    *,
    unit: str,
) -> int:
    """Index of the last markable item at or before the end of the prefix.

    Walks backwards past trailing items that cannot carry a breakpoint (an
    ``assistant`` reply, a ``tool`` result, a ``function_call``). Returns ``-1``
    when there is none.

    Raises:
        ValueError: if ``volatile_tail`` is negative.
    """
    if volatile_tail < 0:
        raise ValueError(f'volatile_tail must be >= 0, got {volatile_tail}')
    index = len(items) - 1 - volatile_tail
    while index >= 0 and not is_markable(items[index]):
        index -= 1
    if index < 0 and items:
        logger.warning(
            'No cacheable {} in the prompt-cache prefix ({} {}s, volatile_tail={}): the transcript '
            'is re-encoded on every turn.',
            unit,
            len(items),
            unit,
            volatile_tail,
        )
    return index


def _is_markable(message: dict[str, Any]) -> bool:
    content = message.get('content')
    return message.get('role') in _MARKABLE_ROLES and isinstance(content, str) and bool(content)


def apply_cache_breakpoints(messages: list[dict[str, Any]], *, volatile_tail: int) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with breakpoints on the system + prefix end.

    Two of the four allowed breakpoints, deduped to one when the system message
    *is* the prefix end:

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

    The leading ``system`` breakpoint ignores ``volatile_tail`` deliberately: the
    system message is built by the framework from a stable prompt, not by the
    caller per turn, so it is a valid read even when every other message is
    volatile.

    Returns ``messages`` unchanged when the total text is below
    `CACHE_MIN_PROMPT_TOKENS` — the provider would cache nothing and the write
    premium would be pure loss.

    A message is skipped when its role is not ``system``/``user`` or its content
    is not a non-empty string — a caller that already built content blocks owns
    its own breakpoints, and re-wrapping would clobber them.

    Raises:
        ValueError: if ``volatile_tail`` is negative.
    """
    out = list(messages)
    prefix_end = _prefix_index(out, volatile_tail, _is_markable, unit='message')
    if not _clears_minimum(out):
        logger.debug(
            'Prompt below the {}-token cache minimum ({} messages) — no breakpoint placed.',
            CACHE_MIN_PROMPT_TOKENS,
            len(out),
        )
        return out
    leading = 0 if out and out[0].get('role') == 'system' else -1
    # A set, not a pair: dedupes when the prefix end *is* the system message, and
    # empties out on an empty list.
    for index in {i for i in (leading, prefix_end) if 0 <= i < len(out) and _is_markable(out[i])}:
        out[index] = {
            **out[index],
            'content': [{'type': 'text', 'text': out[index]['content'], 'cache_control': {'type': 'ephemeral'}}],
        }
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


def mark_responses_input(input_items: list[dict[str, Any]], *, volatile_items: int) -> list[dict[str, Any]]:
    """Return a copy of a Responses ``input`` list with a positioned breakpoint.

    The Responses `input` is a list of items carrying content parts, exactly like
    a chat message list, and the router honours a **per-item** ``cache_control``
    on the last part of an item. Measured against ``anthropic/claude-sonnet-4-6``
    (``scripts/manual_tests/prompt_cache_responses_probe.py``): marking the end
    of the persisted prefix produces a cache read on the next call, while the
    top-level switch alone produces none, because it marks the end of the whole
    input including a trailing item the caller rebuilt.

    The count is ``volatile_items``, **not** ``volatile_tail``: one `Message`
    can render to several Responses items (an assistant turn with tool calls
    becomes a content item plus one ``function_call`` per call), so a count of
    messages would land the breakpoint inside the rebuilt region. Callers holding
    messages compute the item count with `responses_volatile_items`.

    Preferred over the top-level ``cache_control`` body field, which marks the
    end of the whole input: the two compose, but the top-level one then adds a
    write at the end that nothing reads back.

    Returns ``input_items`` unchanged when the total text is below
    `CACHE_MIN_PROMPT_TOKENS`.

    Raises:
        ValueError: if ``volatile_items`` is negative.
    """
    out = list(input_items)
    index = _prefix_index(out, volatile_items, _has_markable_parts, unit='item')
    if index < 0:
        return out
    if not _clears_minimum(out):
        logger.debug(
            'Responses input below the {}-token cache minimum ({} items) — no breakpoint placed.',
            CACHE_MIN_PROMPT_TOKENS,
            len(out),
        )
        return out
    content = out[index]['content']
    # `messages_to_responses_input` renders a plain user turn as a bare string and
    # only an assistant turn as parts, so both shapes arrive here. A string is
    # promoted to a single `input_text` part, which the API accepts for every
    # non-assistant role (an assistant turn is always already a part list).
    parts: list[dict[str, Any]] = (
        [{'type': 'input_text', 'text': content}] if isinstance(content, str) else list(content)
    )
    parts[-1] = {**parts[-1], 'cache_control': {'type': 'ephemeral'}}
    out[index] = {**out[index], 'content': parts}
    return out


def responses_volatile_items(messages: list[Any], *, volatile_tail: int) -> int:
    """How many Responses ``input`` items the last ``volatile_tail`` messages render to.

    `messages_to_responses_input` is a per-message flat-map, so the item count of
    a suffix is exactly the item count that suffix contributes to the whole. This
    is the only supported way to turn a message count into the ``volatile_items``
    that `mark_responses_input` takes.

    Raises:
        ValueError: if ``volatile_tail`` is negative.
    """
    if volatile_tail < 0:
        raise ValueError(f'volatile_tail must be >= 0, got {volatile_tail}')
    if volatile_tail == 0:
        return 0
    from evaluatorq.openresponses.input_items import messages_to_responses_input

    return len(messages_to_responses_input(messages[-volatile_tail:]))


__all__ = [
    'CACHE_MIN_PROMPT_TOKENS',
    'apply_cache_breakpoints',
    'caching_applies',
    'mark_responses_input',
    'responses_volatile_items',
]
