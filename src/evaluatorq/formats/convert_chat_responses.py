"""Convert between chat messages and Responses items."""

from __future__ import annotations

from typing import Any, cast

from loguru import logger

from evaluatorq.contracts import (
    ContentPart,
    FunctionCall,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    Message,
    StrategyToolCall,
    tool_result_to_text,
)
from evaluatorq.formats._shared import arguments_text, join_text, part_text, tool_arguments
from evaluatorq.formats.chat import ChatConversation
from evaluatorq.formats.responses import ResponsesConversation, walk_items
from evaluatorq.openresponses.input_items import messages_to_responses_input, responses_function_call_item_id

_TEXT_PART_TYPES = frozenset({'input_text', 'output_text', 'text', 'summary_text', 'refusal'})
_ROLES = frozenset({'user', 'assistant', 'system', 'developer', 'tool'})


class _State:
    """Mutable bookkeeping for one `responses_to_chat` pass."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.call_names: dict[str, str] = {}
        self.reasoning_dropped = 0
        self.developer_warned = False


def chat_to_responses(conv: ChatConversation) -> ResponsesConversation:
    """Render chat messages as Responses input items."""
    return ResponsesConversation(items=messages_to_responses_input(conv.messages))


def responses_to_chat(conv: ResponsesConversation) -> ChatConversation:
    """Render Responses items as chat messages; see `ResponsesConversation.to_chat` for the losses."""
    state = _State()

    def reasoning(_index: int, _item: dict[str, Any]) -> None:
        state.reasoning_dropped += 1

    handlers = {
        'message': lambda _, item: _message_item(item, state),
        'function_call': lambda _, item: _call_item(item, state),
        'function_call_output': lambda _, item: _output_item(item, state),
        'reasoning': reasoning,
    }
    walk_items(conv.items, handlers, 'chat')
    if state.reasoning_dropped:
        logger.warning('dropped {} reasoning items (chat has no reasoning slot)', state.reasoning_dropped)
    return ChatConversation(messages=state.messages)


def _part(part: Any) -> ContentPart:
    """Map one Responses content part to a chat content part; unknown parts become a `[type]` marker."""
    if not isinstance(part, dict):
        logger.warning('Responses content part is {}, not a dict; rendering a marker.', type(part).__name__)
        return InputTextContent(type='input_text', text=f'[{type(part).__name__}]')
    part_type = part.get('type')
    if part_type in _TEXT_PART_TYPES:
        return InputTextContent(type='input_text', text=part_text(part, 'Responses'))
    if part_type == 'input_image':
        image_url = part.get('image_url')
        return InputImageContent(
            type='input_image',
            image_url=image_url if isinstance(image_url, str) else None,
            file_id=part.get('file_id'),
            detail=part.get('detail') or 'auto',
        )
    if part_type == 'input_file':
        return InputFileContent(
            type='input_file',
            file_id=part.get('file_id'),
            file_data=part.get('file_data'),
            file_url=part.get('file_url'),
            filename=part.get('filename'),
            mime_type=part.get('mime_type'),
        )
    logger.warning('Responses content part type {!r} has no chat equivalent; rendering a marker.', part_type)
    return InputTextContent(type='input_text', text=f'[{part_type}]')


def _content(raw: Any) -> str | list[ContentPart] | None:
    if raw is None or isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        logger.warning('Responses message content is {}; dropping it.', type(raw).__name__)
        return None
    parts = [_part(p) for p in raw]
    if all(isinstance(p, InputTextContent) for p in parts):
        return join_text(cast('InputTextContent', p).text for p in parts)
    return parts


def _message_item(item: dict[str, Any], state: _State) -> None:
    role = item.get('role')
    if role not in _ROLES:
        logger.warning('Skipping Responses message with unknown role {!r}.', role)
        return
    if role == 'developer':
        if not state.developer_warned:
            logger.warning('developer role mapped to system')
            state.developer_warned = True
        role = 'system'
    state.messages.append(Message(role=role, content=_content(item.get('content'))))


def _call_item(item: dict[str, Any], state: _State) -> None:
    call_id = item.get('call_id')
    name = item.get('name')
    if not isinstance(call_id, str) or not call_id:
        logger.warning('Responses function_call {!r} has no call_id; skipping it.', name)
        return
    arguments = item.get('arguments')
    if not isinstance(arguments, str):
        arguments = arguments_text(tool_arguments(arguments, name))
    call = StrategyToolCall(
        id=call_id,
        function=FunctionCall(name=str(name or ''), arguments=arguments),
        item_id=responses_function_call_item_id(item.get('id') if isinstance(item.get('id'), str) else None),
    )
    state.call_names[call_id] = call.function.name
    previous = state.messages[-1] if state.messages else None
    if previous is not None and previous.role == 'assistant':
        previous.tool_calls = [*(previous.tool_calls or []), call]
    else:
        state.messages.append(Message(role='assistant', content=None, tool_calls=[call]))


def _output_item(item: dict[str, Any], state: _State) -> None:
    call_id = item.get('call_id')
    output = item.get('output')
    content = output if isinstance(output, str) else tool_result_to_text(output)
    state.messages.append(
        Message(
            role='tool',
            tool_call_id=call_id if isinstance(call_id, str) else None,
            name=state.call_names.get(call_id) if isinstance(call_id, str) else None,
            content=content,
        )
    )
