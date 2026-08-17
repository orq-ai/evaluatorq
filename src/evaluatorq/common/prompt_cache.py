"""Anthropic prompt-cache breakpoints for multi-turn conversations.

Anthropic caching is **opt-in**: without a ``cache_control`` breakpoint there is
no caching at all, however byte-stable the prefix is. An append-only transcript
buys nothing on its own — it only makes a breakpoint worth placing.

OpenAI and Gemini 2.0+ cache the prefix automatically and the Orq router
**ignores** ``cache_control`` on them rather than rejecting it, so the markers
are safe to send unconditionally: no per-provider branch here. For the same
reason OpenAI's ``prompt_cache_key`` is not set anywhere — it is a routing hint,
not an enabler, and caching there needs no request change.

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

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {'type': 'ephemeral'}
"""The only cache type Anthropic supports. Default TTL is 5 minutes."""

# Only these carry a plain-text body we can safely re-render as a block list.
# `tool` and tool-calling `assistant` turns are left alone: their content shape
# is load-bearing for the provider and a breakpoint there buys nothing extra.
_MARKABLE_ROLES = frozenset({'system', 'user'})


def cached_text_block(text: str) -> dict[str, Any]:
    """Render ``text`` as a text content block carrying a cache breakpoint."""
    return {'type': 'text', 'text': text, 'cache_control': CACHE_CONTROL_EPHEMERAL}


def apply_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with breakpoints on the system + last message.

    Two of the four allowed breakpoints:

    - the leading ``system`` message — static for the whole conversation, so it
      is a cache read on every turn after the first;
    - the final message — the end of the append-only prefix, so the *next* turn
      (which appends past it) reads the whole transcript back.

    A message is skipped when its role is not ``system``/``user`` or its content
    is not a non-empty string — a caller that already built content blocks owns
    its own breakpoints, and re-wrapping would clobber them.
    """
    out = list(messages)
    for index in dict.fromkeys((0, len(out) - 1)):  # dedupe; single-message lists mark once
        if index < 0:
            continue
        message = out[index]
        role = message.get('role')
        content = message.get('content')
        if role not in _MARKABLE_ROLES or not isinstance(content, str) or not content:
            continue
        out[index] = {**message, 'content': [cached_text_block(content)]}
    return out


def responses_cache_body() -> dict[str, Any]:
    """``extra_body`` fragment enabling automatic caching on the Responses API.

    The Responses endpoint accepts a **top-level** ``cache_control``, which marks
    the last cacheable block for you — the closest thing to a conversation-global
    switch, and the reason the Responses path needs no per-message rewriting.
    Router-specific, so callers gate it on ``client_routes_through_orq``.
    """
    return {'cache_control': CACHE_CONTROL_EPHEMERAL}
