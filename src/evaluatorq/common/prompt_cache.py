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


def apply_cache_breakpoints(messages: list[dict[str, Any]], *, volatile_tail: int = 0) -> list[dict[str, Any]]:
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


def responses_cache_body() -> dict[str, Any]:
    """``extra_body`` fragment that enables caching on the Responses API.

    Verified live against ``anthropic/claude-sonnet-4-6`` on the router with a
    cold (uuid-salted) prefix: without it, ``cache_creation_tokens`` is 0 on
    every call; with it, call 1 writes 14,416 tokens and an identical call 2
    reads all 14,416 back, and a call that *appends* to the transcript reads
    14,416 and writes only the 11 new tokens.

    **It marks the end of the whole input — there is no way to position it.**
    A caller that rewrites its trailing message every turn (rather than
    appending) therefore gets a write on every call and a read on none: the
    previous write ended with a message that is no longer a prefix. On the Chat
    Completions path `apply_cache_breakpoints` solves this with
    ``volatile_tail``; here the only fix is to append and never rewrite.

    Router-specific, so callers gate it on ``client_routes_through_orq``.
    """
    return {'cache_control': dict(CACHE_CONTROL_EPHEMERAL)}
