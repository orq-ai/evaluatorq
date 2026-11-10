"""Translate between OpenAI Chat Completions shapes and Anthropic Messages shapes.

Pure functions: no client, no network. ``build_messages_request`` turns the keyword arguments of ``chat.completions.create`` into the keyword arguments of ``messages.create``; ``to_chat_completion`` turns the resulting ``anthropic.types.Message`` into a genuine ``openai.types.chat.ChatCompletion``.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import Function

if TYPE_CHECKING:
    from anthropic.types import Message

DEFAULT_MAX_TOKENS = 16000
"""``max_tokens`` sent when the caller passed none; Anthropic requires one."""

PROVIDER_PREFIX = 'anthropic/'
ORQ_ONLY_EXTRA_BODY_KEYS = frozenset({'retry', 'thread', 'memory'})
"""``extra_body`` keys only the Orq router understands; Anthropic would reject them."""

REFUSAL_FALLBACK_TEXT = 'The model declined to answer this request.'

_FINISH_REASONS: dict[str, str] = {
    'end_turn': 'stop',
    'stop_sequence': 'stop',
    'pause_turn': 'stop',
    'max_tokens': 'length',
    'tool_use': 'tool_calls',
    'refusal': 'content_filter',
}

_EFFORT_LEVELS: dict[str, str | None] = {
    'none': None,
    'minimal': 'low',
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
    'xhigh': 'max',
    'max': 'max',
}
"""OpenAI ``reasoning_effort`` to Anthropic ``output_config.effort``, which allows ``low|medium|high|max``; ``None`` omits it."""

_SUPPORTED_KEYS = frozenset({
    'model',
    'messages',
    'tools',
    'tool_choice',
    'max_tokens',
    'max_completion_tokens',
    'temperature',
    'top_p',
    'reasoning_effort',
    'metadata',
    'extra_headers',
    'extra_body',
    'timeout',
})


def normalize_model(model: str) -> str:
    """Return the bare Claude model id, stripping an ``anthropic/`` prefix.

    Any other ``provider/`` prefix is a configuration mistake (an OpenAI default model reaching an Anthropic client), so it fails loudly instead of reaching the API as a 404. Bedrock inference-profile ARNs contain a slash and are passed through.
    """
    if model.startswith(PROVIDER_PREFIX):
        return model[len(PROVIDER_PREFIX) :]
    if '/' in model and not model.startswith('arn:'):
        raise ValueError(
            f'An Anthropic client needs a Claude model id, got {model!r}. '
            'Set the model explicitly, for example through the EVALUATORQ_FAST_MODEL, EVALUATORQ_SMART_MODEL, '
            'EVALUATORQ_CLASSIFIER_MODEL or EVALUATORQ_EMBEDDING_MODEL role variables, to a name such as '
            '"claude-sonnet-5-5" or "anthropic/claude-sonnet-5-5".'
        )
    return model


def _text_part(part: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {'type': 'text', 'text': part.get('text', '')}
    if part.get('cache_control') is not None:
        block['cache_control'] = part['cache_control']
    return block


def _content_blocks(content: Any, *, role: str) -> list[dict[str, Any]]:
    """Convert message ``content`` (string or parts) into Anthropic text blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        part_type = part.get('type') if isinstance(part, dict) else getattr(part, 'type', None)
        if part_type == 'text':
            block = _text_part(part)
            if block['text']:
                blocks.append(block)
        elif part_type == 'refusal' and role == 'assistant':
            blocks.append({'type': 'text', 'text': part.get('refusal', '')})
        else:
            raise ValueError(
                f'The Anthropic adapter cannot convert a {part_type!r} content part in a {role} message; only text parts are supported.'
            )
    return blocks


def _plain_text(content: Any) -> str:
    return ''.join(block['text'] for block in _content_blocks(content, role='tool'))


def _decode_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get('function') or {}).get('arguments') or ''
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Tool call {call.get("id")!r} has arguments that are not valid JSON: {exc}') from exc
    if not isinstance(decoded, dict):
        raise TypeError(f'Tool call {call.get("id")!r} has arguments that are not a JSON object.')
    return decoded


def _merge_same_role(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join adjacent turns of the same role into one, keeping block order (tool results come first in a user turn)."""
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]['role'] == turn['role']:
            merged[-1] = {'role': turn['role'], 'content': [*merged[-1]['content'], *turn['content']]}
        else:
            merged.append(turn)
    return merged


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Split OpenAI messages into Anthropic ``(system, messages)``.

    ``system`` and ``developer`` messages move to ``system``; ``tool`` messages become ``tool_result`` blocks, with consecutive results merged into a single user turn.
    """
    system_blocks: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            out.append({'role': 'user', 'content': list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message.get('role')
        if role == 'tool':
            pending_results.append({
                'type': 'tool_result',
                'tool_use_id': message.get('tool_call_id'),
                'content': _plain_text(message.get('content')),
            })
            continue
        flush_results()
        if role in ('system', 'developer'):
            system_blocks.extend(_content_blocks(message.get('content'), role=role))
        elif role == 'user':
            blocks = _content_blocks(message.get('content'), role='user')
            if blocks:
                out.append({'role': 'user', 'content': blocks})
            else:
                logger.debug('Dropping a user message with empty content before sending it to Anthropic.')
        elif role == 'assistant':
            blocks = _content_blocks(message.get('content'), role='assistant')
            for call in message.get('tool_calls') or []:
                blocks.append({
                    'type': 'tool_use',
                    'id': call['id'],
                    'name': call['function']['name'],
                    'input': _decode_arguments(call),
                })
            if blocks:
                out.append({'role': 'assistant', 'content': blocks})
            else:
                logger.debug('Dropping an assistant message with empty content before sending it to Anthropic.')
        else:
            raise ValueError(f'The Anthropic adapter cannot convert a message with role {role!r}.')
    flush_results()
    out = _merge_same_role(out)

    system: str | list[dict[str, Any]] | None
    if not system_blocks:
        system = None
    elif any('cache_control' in block for block in system_blocks):
        system = system_blocks
    else:
        system = '\n\n'.join(block['text'] for block in system_blocks)
    return system, out


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function tools to Anthropic ``{name, description, input_schema}`` tools."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get('type') != 'function':
            raise ValueError(f'The Anthropic adapter only supports function tools, got {tool.get("type")!r}.')
        function = tool['function']
        item: dict[str, Any] = {
            'name': function['name'],
            'input_schema': function.get('parameters') or {'type': 'object', 'properties': {}},
        }
        if function.get('description'):
            item['description'] = function['description']
        converted.append(item)
    return converted


def convert_tool_choice(tool_choice: Any) -> dict[str, str] | None:
    """Map ``'auto'`` and ``'none'``; refuse a forced choice, which current Claude models reject."""
    if tool_choice is None:
        return None
    if tool_choice in ('auto', 'none'):
        return {'type': tool_choice}
    raise ValueError(
        f'The Anthropic adapter cannot send tool_choice={tool_choice!r}: a forced or named tool choice returns '
        'HTTP 400 on current Claude models. Use "auto" and name the tool in the prompt instead.'
    )


def build_messages_request(params: dict[str, Any]) -> dict[str, Any]:
    """Turn ``chat.completions.create`` keyword arguments into ``messages.create`` keyword arguments.

    Raises:
        TypeError: for a keyword the adapter does not translate, so nothing is silently ignored.
        ValueError: for content or a tool choice Anthropic cannot represent.
    """
    unknown = sorted(set(params) - _SUPPORTED_KEYS)
    if unknown:
        raise TypeError(f'The Anthropic adapter does not support chat.completions.create parameters: {unknown}')

    system, messages = convert_messages(params['messages'])
    if not messages:
        raise ValueError(
            'The Anthropic adapter has no user or assistant content to send: every message was empty or a system message.'
        )
    max_tokens = params.get('max_completion_tokens')
    if max_tokens is None:
        max_tokens = params.get('max_tokens')
    request: dict[str, Any] = {
        'model': normalize_model(params['model']),
        'max_tokens': max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        'messages': messages,
    }
    if system is not None:
        request['system'] = system
    if params.get('tools'):
        request['tools'] = convert_tools(params['tools'])
        choice = convert_tool_choice(params.get('tool_choice'))
        if choice is not None:
            request['tool_choice'] = choice
    elif params.get('tool_choice') not in (None, 'auto', 'none'):
        convert_tool_choice(params['tool_choice'])
    for key in ('temperature', 'top_p', 'timeout'):
        if params.get(key) is not None:
            request[key] = params[key]
    effort = params.get('reasoning_effort')
    if effort is not None:
        if effort not in _EFFORT_LEVELS:
            raise ValueError(f'reasoning_effort={effort!r} is not supported; use one of {sorted(_EFFORT_LEVELS)}.')
        mapped = _EFFORT_LEVELS[effort]
        if mapped is not None:
            request['output_config'] = {'effort': mapped}
    if params.get('extra_headers'):
        request['extra_headers'] = params['extra_headers']
    extra_body = {k: v for k, v in (params.get('extra_body') or {}).items() if k not in ORQ_ONLY_EXTRA_BODY_KEYS}
    if extra_body:
        request['extra_body'] = extra_body
    return request


def convert_usage(message: Message) -> CompletionUsage:
    """Fold Anthropic's split input counters into OpenAI's ``prompt_tokens`` accounting."""
    usage = message.usage
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    prompt = usage.input_tokens + cache_read + cache_write
    return CompletionUsage.model_validate({
        'prompt_tokens': prompt,
        'completion_tokens': usage.output_tokens,
        'total_tokens': prompt + usage.output_tokens,
        'prompt_tokens_details': {'cached_tokens': cache_read},
        'cache_creation_input_tokens': cache_write,
    })


def to_chat_completion(message: Message) -> ChatCompletion:
    """Build a genuine ``ChatCompletion`` from an Anthropic ``Message``."""
    text = ''.join(block.text for block in message.content if block.type == 'text')
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=block.id,
            type='function',
            function=Function(name=block.name, arguments=json.dumps(block.input)),
        )
        for block in message.content
        if block.type == 'tool_use'
    ]
    finish_reason = _FINISH_REASONS.get(message.stop_reason or 'end_turn', 'stop')
    refusal = None
    if message.stop_reason == 'refusal':
        refusal = text or REFUSAL_FALLBACK_TEXT
        text = ''
    content: str | None = text or (None if tool_calls or refusal else '')
    model = message.model.removeprefix(PROVIDER_PREFIX)
    return ChatCompletion(
        id=message.id,
        object='chat.completion',
        created=int(time.time()),
        model=f'{PROVIDER_PREFIX}{model}',
        choices=[
            Choice(
                index=0,
                finish_reason=cast(
                    'Literal["stop", "length", "tool_calls", "content_filter", "function_call"]', finish_reason
                ),
                message=ChatCompletionMessage(
                    role='assistant',
                    content=content,
                    tool_calls=tool_calls or None,
                    refusal=refusal,
                ),
            )
        ],
        usage=convert_usage(message),
    )
