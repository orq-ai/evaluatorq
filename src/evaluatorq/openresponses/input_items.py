"""Render :class:`Message` transcripts as Responses-API ``input`` items.

The Responses API is not chat-completions: it has no ``role: "tool"`` message and
ignores a ``tool_calls`` key on a message item. A transcript replayed with the
chat-completions shape is rejected outright (``Invalid value: 'tool'``) and, worse,
silently drops the assistant's tool calls. Every stateless target that replays a
transcript against ``/v3/router/responses`` funnels through here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evaluatorq.contracts import Message


def message_to_responses_input_items(m: Message) -> list[dict[str, Any]]:
    """Render a single :class:`Message` as one or more Responses-API input items.

    An assistant turn with tool calls becomes a ``function_call`` item per call
    (plus a leading assistant text message when content is present); a ``tool``
    result becomes a ``function_call_output``; anything else is a plain
    ``{"role", "content"}`` message. This preserves the multi-turn tool context
    that a naive flatten drops, matching what the OpenAI SDK's
    ``to_input_list()`` round-trips.
    """
    if m.role == 'tool':
        return [{'type': 'function_call_output', 'call_id': m.tool_call_id or '', 'output': m.content or ''}]
    if m.role == 'assistant' and m.tool_calls:
        items: list[dict[str, Any]] = []
        if m.content:
            items.append({'role': 'assistant', 'content': m.content})
        for tc in m.tool_calls:
            fc: dict[str, Any] = {
                'type': 'function_call',
                'call_id': tc.id,
                'name': tc.function.name,
                'arguments': tc.function.arguments,
            }
            # Echo the Responses-API item id (fc_*) when available so the
            # function_call item round-trips intact across turns.
            if tc.item_id:
                fc['id'] = tc.item_id
            items.append(fc)
        return items
    # Multi-part content passes straight through as Responses-API content parts.
    if isinstance(m.content, list):
        return [{'role': m.role, 'content': [p.model_dump(mode='json') for p in m.content]}]
    return [{'role': m.role, 'content': m.content or ''}]


def messages_to_responses_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Render a full transcript as the Responses-API ``input`` array."""
    return [item for m in messages for item in message_to_responses_input_items(m)]


__all__ = ['message_to_responses_input_items', 'messages_to_responses_input']
