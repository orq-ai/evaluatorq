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
    # Set, not a pair: dedupes a single-message list and empties out entirely on
    # an empty one, so neither needs its own guard.
    for index in {i for i in (0, len(out) - 1) if 0 <= i < len(out)}:
        message = out[index]
        role = message.get('role')
        content = message.get('content')
        if role not in _MARKABLE_ROLES or not isinstance(content, str) or not content:
            continue
        out[index] = {**message, 'content': [cached_text_block(content)]}
    return out


def responses_cache_body() -> dict[str, Any]:
    """``extra_body`` fragment intended to enable caching on the Responses API.

    **Unverified.** The Orq Responses reference does not document a top-level
    ``cache_control``; the documented placement is on text blocks in router Chat
    Completions. Sent as an unknown body field it is most likely ignored — a
    no-op rather than a cache — so treat the Responses path as uncached until a
    live trace shows a cache read. Router-specific either way, so callers gate it
    on ``client_routes_through_orq``.
    """
    return {'cache_control': dict(CACHE_CONTROL_EPHEMERAL)}
