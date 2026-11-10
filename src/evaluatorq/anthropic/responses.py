"""Translate between OpenAI Responses shapes and Anthropic Messages shapes.

The Responses face of the adapter, used by the simulation judge and the user simulator. ``build_responses_request`` turns the keyword arguments of ``responses.create`` into the keyword arguments of ``messages.create``; ``to_response`` turns the resulting ``anthropic.types.Message`` into a genuine ``openai.types.responses.Response``. Everything that both surfaces share (model ids, effort, schemas, tools, usage, same-role merging) lives in ``convert.py`` and is called from here.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import omit
from openai.lib._parsing._responses import parse_response, type_to_text_format_param
from openai.types.responses import (
    ParsedResponse,
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails

from evaluatorq.anthropic.convert import (
    DEFAULT_MAX_TOKENS,
    PROVIDER_PREFIX,
    REFUSAL_FALLBACK_TEXT,
    adapt_schema,
    apply_shared_params,
    build_output_config,
    content_blocks,
    convert_tool_choice,
    convert_tools,
    decode_arguments,
    merge_same_role,
    normalize_model,
    usage_counts,
)

if TYPE_CHECKING:
    from anthropic.types import Message

_SUPPORTED_KEYS = frozenset({
    'model',
    'input',
    'instructions',
    'tools',
    'tool_choice',
    'temperature',
    'top_p',
    'max_output_tokens',
    'reasoning',
    'text',
    'metadata',
    'extra_headers',
    'extra_body',
    'timeout',
    'fallbacks',
})
_TEXT_TYPES = ('input_text', 'output_text')
_SURFACE = 'responses.create'


def _item_text(output: Any) -> str:
    """Flatten a ``function_call_output.output`` (a string or ``input_text`` parts) into the text of a ``tool_result``."""
    return ''.join(
        block['text'] for block in content_blocks(output, role='tool', text_types=_TEXT_TYPES, surface=_SURFACE)
    )


def convert_input(
    items: str | list[dict[str, Any]], instructions: str | None
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Split Responses ``input`` into Anthropic ``(system, messages)``.

    A string is one user message. ``message`` items become text blocks (``cache_control`` kept), ``function_call`` items become ``tool_use`` blocks, ``function_call_output`` items become ``tool_result`` blocks, and ``reasoning`` items are dropped. ``instructions`` and any ``system`` or ``developer`` item go to ``system``. Adjacent same-role turns are merged exactly as the Chat Completions face does.
    """
    if isinstance(items, str):
        items = [{'role': 'user', 'content': items}]
    system_blocks: list[dict[str, Any]] = []
    if instructions:
        system_blocks.append({'type': 'text', 'text': instructions})
    out: list[dict[str, Any]] = []
    for item in items:
        kind = item.get('type') or ('message' if 'role' in item else None)
        if kind == 'message':
            role = item.get('role')
            blocks = content_blocks(item.get('content'), role=str(role), text_types=_TEXT_TYPES, surface=_SURFACE)
            if role in ('system', 'developer'):
                system_blocks.extend(blocks)
            elif role in ('user', 'assistant'):
                if blocks:
                    out.append({'role': role, 'content': blocks})
                else:
                    logger.debug('Dropping a {} item with empty content before sending it to Anthropic.', role)
            else:
                raise ValueError(f'The Anthropic adapter cannot convert an input message with role {role!r}.')
        elif kind == 'function_call':
            out.append({
                'role': 'assistant',
                'content': [
                    {
                        'type': 'tool_use',
                        'id': item['call_id'],
                        'name': item['name'],
                        'input': decode_arguments(item.get('arguments'), item['call_id']),
                    }
                ],
            })
        elif kind == 'function_call_output':
            out.append({
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': item['call_id'], 'content': _item_text(item.get('output'))}
                ],
            })
        elif kind == 'reasoning':
            continue
        else:
            raise ValueError(f'The Anthropic adapter cannot convert a Responses input item of type {kind!r}.')
    system: str | list[dict[str, Any]] | None
    if not system_blocks:
        system = None
    elif any('cache_control' in block for block in system_blocks):
        system = system_blocks
    else:
        system = '\n\n'.join(block['text'] for block in system_blocks)
    return system, merge_same_role(out)


def _chat_shaped_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap flat Responses function tools in the Chat Completions shape so ``convert_tools`` can convert them."""
    shaped: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get('type') != 'function':
            raise ValueError(f'The Anthropic adapter only supports function tools, got {tool.get("type")!r}.')
        function = {key: tool[key] for key in ('name', 'description', 'parameters') if tool.get(key) is not None}
        shaped.append({'type': 'function', 'function': function})
    return shaped


def _output_format(text: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map ``text.format`` to ``output_config.format``; ``text`` and ``json_object`` send nothing, as on the Chat Completions face."""
    if not text:
        return None
    unknown = sorted(set(text) - {'format'})
    if unknown:
        raise TypeError(f'The Anthropic adapter does not support responses.create text options: {unknown}')
    fmt = text.get('format')
    if fmt is None or fmt.get('type') in ('text', 'json_object'):
        return None
    if fmt.get('type') != 'json_schema':
        raise ValueError(f'The Anthropic adapter does not support text.format type {fmt.get("type")!r}.')
    if fmt.get('schema') is None:
        raise ValueError('text.format of type json_schema needs a schema.')
    return {'type': 'json_schema', 'schema': adapt_schema(fmt['schema'])}


def _effort(reasoning: dict[str, Any] | None) -> str | None:
    if not reasoning:
        return None
    unknown = sorted(set(reasoning) - {'effort', 'summary', 'generate_summary'})
    if unknown:
        raise TypeError(f'The Anthropic adapter does not support responses.create reasoning options: {unknown}')
    return reasoning.get('effort')


def build_responses_request(params: dict[str, Any], *, text_format: Any = None) -> dict[str, Any]:
    """Turn ``responses.create`` keyword arguments into ``messages.create`` keyword arguments.

    ``text_format`` is the Pydantic class of ``responses.parse``; it becomes ``text.format`` through openai's own helper, so both providers see the same schema.

    Raises:
        TypeError: for a keyword the adapter does not translate, so nothing is silently ignored.
        ValueError: for content or a tool choice Anthropic cannot represent.
    """
    unknown = sorted(set(params) - _SUPPORTED_KEYS)
    if unknown:
        raise TypeError(f'The Anthropic adapter does not support responses.create parameters: {unknown}')

    system, messages = convert_input(params['input'], params.get('instructions'))
    if not messages:
        raise ValueError(
            'The Anthropic adapter has no user or assistant content to send: every input item was empty, '
            'a system item or a reasoning item.'
        )
    max_tokens = params.get('max_output_tokens')
    request: dict[str, Any] = {
        'model': normalize_model(params['model']),
        'max_tokens': max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        'messages': messages,
    }
    if system is not None:
        request['system'] = system
    if params.get('tools'):
        request['tools'] = convert_tools(_chat_shaped_tools(params['tools']))
        choice = convert_tool_choice(params.get('tool_choice'))
        if choice is not None:
            request['tool_choice'] = choice
    elif params.get('tool_choice') not in (None, 'auto', 'none'):
        convert_tool_choice(params['tool_choice'])
    text = params.get('text')
    if text_format is not None:
        text = {**(text or {}), 'format': type_to_text_format_param(text_format)}
    apply_shared_params(request, params, build_output_config(_output_format(text), _effort(params.get('reasoning'))))
    return request


def convert_response_usage(message: Message) -> ResponseUsage:
    """Build ``ResponseUsage`` from an Anthropic message's usage, folding cache counters into ``input_tokens``."""
    prompt, completion, cache_read, cache_write = usage_counts(message)
    return ResponseUsage.model_validate({
        'input_tokens': prompt,
        'output_tokens': completion,
        'total_tokens': prompt + completion,
        'input_tokens_details': {'cached_tokens': cache_read},
        'output_tokens_details': {'reasoning_tokens': 0},
        'cache_creation_input_tokens': cache_write,
    })


def to_response(message: Message) -> Response:
    """Build a genuine ``Response`` from an Anthropic ``Message``.

    Text blocks are joined into one ``output_text`` part (a refusal becomes one ``refusal`` part), each ``tool_use`` becomes a ``function_call`` item, and thinking blocks are dropped. ``max_tokens`` gives ``status='incomplete'`` with ``incomplete_details.reason='max_output_tokens'``.
    """
    text = ''.join(block.text for block in message.content if block.type == 'text')
    output: list[Any] = []
    if message.stop_reason == 'refusal':
        output.append(
            ResponseOutputMessage(
                id=message.id,
                type='message',
                role='assistant',
                status='completed',
                content=[ResponseOutputRefusal(type='refusal', refusal=text or REFUSAL_FALLBACK_TEXT)],
            )
        )
    elif text:
        output.append(
            ResponseOutputMessage(
                id=message.id,
                type='message',
                role='assistant',
                status='completed',
                content=[ResponseOutputText(type='output_text', text=text, annotations=[])],
            )
        )
    output.extend(
        ResponseFunctionToolCall(
            type='function_call',
            call_id=block.id,
            name=block.name,
            arguments=json.dumps(block.input),
            status='completed',
        )
        for block in message.content
        if block.type == 'tool_use'
    )
    truncated = message.stop_reason == 'max_tokens'
    model = message.model.removeprefix(PROVIDER_PREFIX)
    return Response(
        id=message.id,
        object='response',
        created_at=float(time.time()),
        model=f'{PROVIDER_PREFIX}{model}',
        output=output,
        status='incomplete' if truncated else 'completed',
        incomplete_details=IncompleteDetails(reason='max_output_tokens') if truncated else None,
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
        usage=convert_response_usage(message),
    )


def to_parsed_response(message: Message, *, text_format: Any, tools: Any = None) -> ParsedResponse[Any]:
    """Build a ``ParsedResponse`` the way ``openai``'s ``responses.parse`` does, by calling its own ``parse_response``.

    Invalid JSON for the class raises ``pydantic.ValidationError``. A refusal leaves ``output_parsed`` as ``None`` and keeps the ``refusal`` part, which ``first_responses_refusal`` reads. A ``max_tokens`` stop is returned as ``incomplete``; ``responses_stop_reason`` reports it as a truncation, as on OpenAI.
    """
    return parse_response(
        text_format=text_format if text_format is not None else omit,
        input_tools=tools or omit,
        response=to_response(message),
    )
