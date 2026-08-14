"""Render `Message` transcripts as Responses-API ``input`` items.

The Responses API is not chat-completions: it has no ``role: "tool"`` message and
ignores a ``tool_calls`` key on a message item. A transcript replayed with the
chat-completions shape is rejected outright (``Invalid value: 'tool'``) and, worse,
silently drops the assistant's tool calls. Every stateless target that replays a
transcript against ``/v3/router/responses`` funnels through here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from evaluatorq.contracts import ContentPart, Message


def _content_parts(content: list[ContentPart]) -> list[dict[str, Any]]:
    """Serialize multi-part content to plain dicts the SDK can encode."""
    return [p.model_dump(mode='json') for p in content]


def _assistant_content(content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
    """Render an assistant turn's content as ``output_text`` parts.

    An assistant item's content MUST be a list of ``output_text`` parts. A bare
    string, or a list of ``input_text`` parts, is **silently dropped** by the Orq
    router — the model receives a transcript with no assistant turns in it at all
    (some backends 400 instead: "cannot unmarshal string into Go struct field
    messageItemRaw.content"). That drop is why a simulation judge reported "the
    agent has not yet responded" and no criterion about agent behaviour could
    fail (RES-1308). Only text is representable here; there is no ``output_image``.
    """
    if content is None:
        return [{'type': 'output_text', 'text': ''}]
    if isinstance(content, str):
        return [{'type': 'output_text', 'text': content}]
    parts: list[dict[str, Any]] = []
    dropped: list[str] = []
    for p in content:
        text = getattr(p, 'text', None)
        if isinstance(text, str):
            parts.append({'type': 'output_text', 'text': text})
        else:
            dropped.append(p.type)
    if dropped:
        # loguru formats with {}, not %s.
        logger.warning(
            'Dropping non-text part(s) {} from an assistant turn in the Responses input: '
            'assistant content only supports output_text. The model will not see them.',
            ', '.join(dropped),
        )
    return parts or [{'type': 'output_text', 'text': ''}]


def _tool_output(content: str | list[ContentPart] | None) -> str | list[dict[str, Any]]:
    """Render tool-result content for ``function_call_output.output``.

    That field takes a string *or* a list of ``input_text``/``input_image``/
    ``input_file`` parts — exactly the three members of `ContentPart` — so
    multi-part content passes straight through instead of being flattened to
    text. Flattening would silently discard an image or file a tool returned.
    """
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return _content_parts(content)


def message_to_responses_input_items(m: Message) -> list[dict[str, Any]]:
    """Render a single `Message` as one or more Responses-API input items.

    An assistant turn with tool calls becomes a ``function_call`` item per call
    (plus a leading assistant text message when content is present); a ``tool``
    result becomes a ``function_call_output``; anything else is a plain
    ``{"role", "content"}`` message. This preserves the multi-turn tool context
    that a naive flatten drops, matching what the OpenAI SDK's
    ``to_input_list()`` round-trips.
    """
    if m.role == 'tool':
        # A function_call_output must reference a prior function_call by a
        # non-empty call_id; the API rejects "" outright. An orphaned tool
        # result (no tool_call_id) is unreferenceable, so drop it rather than
        # fail the whole turn on one malformed row — but say so, because a tool
        # result vanishing from replayed history is otherwise invisible.
        if not m.tool_call_id:
            logger.warning(
                'Dropping a tool message with no tool_call_id from the Responses input: '
                'a function_call_output cannot be sent without a call_id to reference. '
                'The model will not see this tool result.'
            )
            return []
        return [{'type': 'function_call_output', 'call_id': m.tool_call_id, 'output': _tool_output(m.content)}]
    if m.role == 'assistant' and m.tool_calls:
        items: list[dict[str, Any]] = []
        if m.content:
            items.append({'role': 'assistant', 'content': _assistant_content(m.content)})
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
    # Assistant turns need output_text parts; every other role takes input_* parts
    # (or a bare string) as-is.
    if m.role == 'assistant':
        return [{'role': 'assistant', 'content': _assistant_content(m.content)}]
    if isinstance(m.content, list):
        return [{'role': m.role, 'content': _content_parts(m.content)}]
    return [{'role': m.role, 'content': m.content or ''}]


def messages_to_responses_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Render a full transcript as the Responses-API ``input`` array."""
    return [item for m in messages for item in message_to_responses_input_items(m)]


__all__ = ['message_to_responses_input_items', 'messages_to_responses_input']
