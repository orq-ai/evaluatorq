"""Translate between OpenAI Chat Completions shapes and Anthropic Messages shapes.

Pure functions: no client, no network. ``build_messages_request`` turns the keyword arguments of ``chat.completions.create`` into the keyword arguments of ``messages.create``; ``to_chat_completion`` turns the resulting ``anthropic.types.Message`` into a genuine ``openai.types.chat.ChatCompletion``.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger
from openai import omit
from openai.lib._parsing import parse_chat_completion, type_to_response_format_param
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ParsedChatCompletion,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import Function

from evaluatorq.common.llm_client import LLMConfigurationError, ModelClientMismatchError

if TYPE_CHECKING:
    from anthropic.types import Message

DEFAULT_MAX_TOKENS = 16000
"""``max_tokens`` sent when the caller passed none; Anthropic requires one."""

PROVIDER_PREFIX = 'anthropic/'
ORQ_ONLY_EXTRA_BODY_KEYS = frozenset({'retry', 'thread', 'memory'})
"""``extra_body`` keys only the Orq router understands; Anthropic would reject them."""
ORQ_ONLY_PARAMS = frozenset({'metadata'})
"""Top-level call parameters only the Orq router reads (run metadata); accepted and dropped, with a debug line."""

REFUSAL_FALLBACK_TEXT = 'The model declined to answer this request.'

LEADING_USER_TEXT = '(The conversation starts here.)'
"""Text of the user turn put in front of a conversation that starts with an assistant turn; the Messages API needs a user turn first."""


class UnsupportedParameterError(LLMConfigurationError, TypeError):
    """A request parameter the adapter does not translate. A `TypeError` too, as an unknown keyword argument would be."""


_FINISH_REASONS: dict[str, str] = {
    'end_turn': 'stop',
    'stop_sequence': 'stop',
    'pause_turn': 'stop',
    'max_tokens': 'length',
    'model_context_window_exceeded': 'length',
    'tool_use': 'tool_calls',
    'refusal': 'content_filter',
}

TRUNCATING_STOP_REASONS = frozenset({'max_tokens', 'model_context_window_exceeded'})
"""Stop reasons where the reply was cut off before the model finished."""

_unknown_stop_reasons_logged: set[str] = set()


def finish_reason_for(stop_reason: str | None) -> str:
    """OpenAI ``finish_reason`` for an Anthropic ``stop_reason``; an unknown one is logged once and read as ``stop``."""
    reason = stop_reason or 'end_turn'
    mapped = _FINISH_REASONS.get(reason)
    if mapped is not None:
        return mapped
    if reason not in _unknown_stop_reasons_logged:
        _unknown_stop_reasons_logged.add(reason)
        logger.warning(
            'The Anthropic adapter does not know stop_reason {!r}; reporting it as finish_reason "stop". '
            'A reply cut off for this reason would look complete.',
            reason,
        )
    return 'stop'


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
    'extra_headers',
    'extra_body',
    'timeout',
    'response_format',
    'fallbacks',
})


_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    'minimum',
    'maximum',
    'exclusiveMinimum',
    'exclusiveMaximum',
    'multipleOf',
    'minLength',
    'maxLength',
    'maxItems',
    'uniqueItems',
})
"""JSON Schema keywords Anthropic structured outputs reject: numerical, string and complex array constraints. ``minItems`` is kept only for 0 and 1."""


def adapt_schema(schema: Any) -> Any:
    """Copy a JSON schema, dropping the keywords Anthropic structured outputs reject and closing every object.

    Only schema positions are walked (``properties`` values, ``items``, ``prefixItems``, schema-valued ``additionalProperties``, ``not``, ``anyOf``/``allOf``/``oneOf`` members, ``$defs``), so a property that happens to be named ``minimum`` survives. Objects without ``additionalProperties`` get ``False``, which Anthropic requires.
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
            continue
        if key == 'minItems' and value not in (0, 1):
            continue
        if key in ('properties', '$defs', 'definitions') and isinstance(value, dict):
            out[key] = {name: adapt_schema(sub) for name, sub in value.items()}
        elif key in ('anyOf', 'allOf', 'oneOf', 'prefixItems') and isinstance(value, list):
            out[key] = [adapt_schema(sub) for sub in value]
        elif key in ('items', 'additionalProperties', 'not'):
            out[key] = adapt_schema(value)
        else:
            out[key] = value
    if out.get('type') == 'object' and 'additionalProperties' not in out:
        out['additionalProperties'] = False
    return out


def response_format_to_output_format(response_format: Any) -> dict[str, Any] | None:
    """Map a ``response_format`` to ``output_config.format``, or ``None`` when a plain call is right.

    A Pydantic class is turned into a schema by openai's own helper (`type_to_response_format_param`), so both providers see the same schema. ``{'type': 'json_object'}`` and ``{'type': 'text'}`` send nothing: Anthropic has no JSON mode, and the callers parse the text themselves.
    """
    if response_format is None:
        return None
    param = cast(
        'dict[str, Any]',
        response_format if isinstance(response_format, dict) else type_to_response_format_param(response_format),
    )
    kind = param.get('type')
    if kind in ('json_object', 'text'):
        return None
    if kind != 'json_schema':
        raise ValueError(f'The Anthropic adapter does not support response_format type {kind!r}.')
    schema = param['json_schema'].get('schema')
    if schema is None:
        raise ValueError('response_format of type json_schema needs json_schema.schema.')
    return {'type': 'json_schema', 'schema': adapt_schema(schema)}


def _role_variables_for(model: str) -> str:
    """The role variable(s) that currently resolve to ``model``; the non-embedding role variables plus the per-call ``model=`` argument when none does (the model was set per call)."""
    from evaluatorq.common.model_roles import ROLE_ENV, ROLES, role_model

    names = [ROLE_ENV[role] for role in ROLES if role_model(role) == model]
    if not names:
        # Embedding is left out: an embedding model never reaches a Claude chat call.
        names = [env for role, env in ROLE_ENV.items() if role != 'embedding']
        return 'the model= argument of the call, or ' + (
            'the ' + ', '.join(names[:-1]) + f' or {names[-1]} role variables'
        )
    return (
        f'the {names[0]} role variable'
        if len(names) == 1
        else 'the ' + ', '.join(names[:-1]) + f' or {names[-1]} role variables'
    )


def normalize_model(model: str) -> str:
    """Return the bare Claude model id, stripping an ``anthropic/`` prefix.

    Any other ``provider/`` prefix is a configuration mistake (an OpenAI default model reaching an Anthropic client), so it raises `ModelClientMismatchError` instead of reaching the API as a 404. Bedrock inference-profile ARNs contain a slash and are passed through.
    """
    if model.startswith(PROVIDER_PREFIX):
        return model[len(PROVIDER_PREFIX) :]
    if '/' in model and not model.startswith('arn:'):
        raise ModelClientMismatchError(
            f'An Anthropic client needs a Claude model id, got {model!r}. '
            f'Set the model explicitly, for example through {_role_variables_for(model)}, '
            'to a name such as "claude-sonnet-5-5" or "anthropic/claude-sonnet-5-5".'
        )
    return model


def _part_dict(part: Any) -> dict[str, Any]:
    """A content part as a dict: a dict as is, a Pydantic part (such as ``openai.types``) through ``model_dump``, any other object through its attributes."""
    if isinstance(part, dict):
        return part
    dump = getattr(part, 'model_dump', None)
    if callable(dump):
        return cast('dict[str, Any]', dump(exclude_none=True))
    return {
        key: getattr(part, key)
        for key in ('type', 'text', 'refusal', 'cache_control')
        if getattr(part, key, None) is not None
    }


def _text_part(part: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {'type': 'text', 'text': part.get('text', '')}
    if part.get('cache_control') is not None:
        block['cache_control'] = part['cache_control']
    return block


def content_blocks(
    content: Any, *, role: str, text_types: tuple[str, ...] = ('text',), surface: str = 'chat.completions.create'
) -> list[dict[str, Any]]:
    """Convert message ``content`` (string or parts) into Anthropic text blocks.

    ``text_types`` names the part types that carry text: ``text`` for Chat Completions, ``input_text`` and ``output_text`` for Responses.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    blocks: list[dict[str, Any]] = []
    for raw_part in content:
        part = _part_dict(raw_part)
        part_type = part.get('type')
        if part_type in text_types:
            block = _text_part(part)
            if block['text']:
                blocks.append(block)
        elif part_type == 'refusal' and role == 'assistant':
            blocks.append({'type': 'text', 'text': part.get('refusal', '')})
        else:
            raise ValueError(
                f'The Anthropic adapter cannot convert a {part_type!r} content part in a {role} message of {surface}; only text parts are supported.'
            )
    return blocks


def _plain_text(content: Any) -> str:
    return ''.join(block['text'] for block in content_blocks(content, role='tool'))


def decode_arguments(raw: str | None, call_id: Any) -> dict[str, Any]:
    """Decode a tool call's JSON ``arguments`` string into the object Anthropic's ``tool_use.input`` takes."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Tool call {call_id!r} has arguments that are not valid JSON: {exc}') from exc
    if not isinstance(decoded, dict):
        raise TypeError(f'Tool call {call_id!r} has arguments that are not a JSON object.')
    return decoded


def merge_same_role(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join adjacent turns of the same role into one, keeping block order (tool results come first in a user turn)."""
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]['role'] == turn['role']:
            merged[-1] = {'role': turn['role'], 'content': [*merged[-1]['content'], *turn['content']]}
        else:
            merged.append(turn)
    return merged


def order_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent same-role turns, then put a short user turn in front when the conversation starts with an assistant turn.

    The Messages API needs the first message to be a user turn. A conversation can legitimately start with an assistant turn: from its second turn on, the simulated user's request is the transcript with roles swapped, so its first message is the simulator's own opening line.
    """
    merged = merge_same_role(turns)
    if merged and merged[0]['role'] == 'assistant':
        merged.insert(0, {'role': 'user', 'content': [{'type': 'text', 'text': LEADING_USER_TEXT}]})
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
            system_blocks.extend(content_blocks(message.get('content'), role=role))
        elif role == 'user':
            blocks = content_blocks(message.get('content'), role='user')
            if blocks:
                out.append({'role': 'user', 'content': blocks})
            else:
                logger.debug('Dropping a user message with empty content before sending it to Anthropic.')
        elif role == 'assistant':
            blocks = content_blocks(message.get('content'), role='assistant')
            for call in message.get('tool_calls') or []:
                blocks.append({
                    'type': 'tool_use',
                    'id': call['id'],
                    'name': call['function']['name'],
                    'input': decode_arguments((call.get('function') or {}).get('arguments'), call.get('id')),
                })
            if blocks:
                out.append({'role': 'assistant', 'content': blocks})
            else:
                logger.debug('Dropping an assistant message with empty content before sending it to Anthropic.')
        else:
            raise ValueError(f'The Anthropic adapter cannot convert a message with role {role!r}.')
    flush_results()
    return build_system(system_blocks), order_turns(out)


def build_system(system_blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]] | None:
    """Build Anthropic ``system``: ``None`` when empty, the block list when any block has ``cache_control``, else one ``\\n\\n``-joined string."""
    if not system_blocks:
        return None
    if any('cache_control' in block for block in system_blocks):
        return system_blocks
    return '\n\n'.join(block['text'] for block in system_blocks)


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert function tools to Anthropic ``{name, description, input_schema}`` tools.

    Takes both shapes: Chat Completions nests the definition under ``function``, Responses puts ``name``, ``description`` and ``parameters`` on the tool itself.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get('type') != 'function':
            raise ValueError(f'The Anthropic adapter only supports function tools, got {tool.get("type")!r}.')
        function = tool.get('function', tool)
        item: dict[str, Any] = {
            'name': function['name'],
            'input_schema': function.get('parameters') or {'type': 'object', 'properties': {}},
        }
        if function.get('description'):
            item['description'] = function['description']
        converted.append(item)
    return converted


def convert_tool_choice(tool_choice: Any) -> dict[str, str] | None:
    """Map an OpenAI ``tool_choice`` to Anthropic's.

    ``'auto'`` and ``'none'`` keep their names (Anthropic has both), ``'required'`` becomes ``{'type': 'any'}``, and a named function becomes ``{'type': 'tool', 'name': ...}``. The named form is accepted in the Chat Completions shape (``{'type': 'function', 'function': {'name': ...}}``) and the Responses shape (``{'type': 'function', 'name': ...}``). Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 and later answer a forced choice (``any`` or ``tool``) with HTTP 400; other Claude models accept it, so the adapter translates it and leaves the decision to the API.
    """
    if tool_choice is None:
        return None
    if tool_choice in ('auto', 'none'):
        return {'type': tool_choice}
    if tool_choice == 'required':
        return {'type': 'any'}
    if isinstance(tool_choice, dict) and tool_choice.get('type') == 'function':
        function = tool_choice.get('function')
        name = function.get('name') if isinstance(function, dict) else tool_choice.get('name')
        if isinstance(name, str) and name:
            return {'type': 'tool', 'name': name}
    raise ValueError(f'The Anthropic adapter cannot translate tool_choice={tool_choice!r}.')


def apply_tools(request: dict[str, Any], tools: list[dict[str, Any]] | None, tool_choice: Any) -> None:
    """Set ``tools`` and ``tool_choice`` on an Anthropic request from Chat- or Responses-shaped ``tools``.

    A ``tool_choice`` without tools is not sent (Anthropic has nothing to choose from), but it is still checked, so an untranslatable value fails the same way with or without tools.
    """
    choice = convert_tool_choice(tool_choice)
    if tools:
        request['tools'] = convert_tools(tools)
        if choice is not None:
            request['tool_choice'] = choice


def build_output_config(output_format: dict[str, Any] | None, effort: str | None) -> dict[str, Any]:
    """Build ``output_config`` from an ``output_config.format`` (or ``None``) and an OpenAI reasoning effort (the ``effort`` key)."""
    output_config: dict[str, Any] = {}
    if output_format is not None:
        output_config['format'] = output_format
    if effort is not None:
        if effort not in _EFFORT_LEVELS:
            raise ValueError(f'reasoning_effort={effort!r} is not supported; use one of {sorted(_EFFORT_LEVELS)}.')
        mapped = _EFFORT_LEVELS[effort]
        if mapped is not None:
            output_config['effort'] = mapped
        else:
            logger.debug(
                'The Anthropic adapter does not send reasoning_effort={!r}: Anthropic has no such level.', effort
            )
    return output_config


def build_messages_request(params: dict[str, Any]) -> dict[str, Any]:
    """Turn ``chat.completions.create`` keyword arguments into ``messages.create`` keyword arguments.

    Raises:
        UnsupportedParameterError: for a keyword the adapter does not translate, so nothing is silently ignored. It is both a `TypeError` and an `LLMConfigurationError`.
        ValueError: for content or a tool choice Anthropic cannot represent.
    """
    unknown = sorted(set(params) - _SUPPORTED_KEYS - ORQ_ONLY_PARAMS)
    if unknown:
        raise UnsupportedParameterError(
            f'The Anthropic adapter does not support chat.completions.create parameters: {unknown}'
        )

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
    apply_tools(request, params.get('tools'), params.get('tool_choice'))
    apply_shared_params(
        request,
        params,
        build_output_config(
            response_format_to_output_format(params.get('response_format')), params.get('reasoning_effort')
        ),
    )
    return request


def apply_shared_params(request: dict[str, Any], params: dict[str, Any], output_config: dict[str, Any]) -> None:
    """Copy the keys both surfaces send alike into ``request``.

    ``temperature``, ``top_p`` and ``timeout`` only when given; ``extra_headers`` as is; ``extra_body`` without the Orq-only keys, plus ``fallbacks`` (which Anthropic takes only in the body); ``output_config`` when it has content.
    """
    for key in sorted(ORQ_ONLY_PARAMS & set(params)):
        logger.debug('The Anthropic adapter does not send {!r}: only the Orq router reads it.', key)
    for key in ('temperature', 'top_p', 'timeout'):
        if params.get(key) is not None:
            request[key] = params[key]
    if output_config:
        request['output_config'] = output_config
    if params.get('extra_headers'):
        request['extra_headers'] = params['extra_headers']
    extra_body = {k: v for k, v in (params.get('extra_body') or {}).items() if k not in ORQ_ONLY_EXTRA_BODY_KEYS}
    for key in sorted(ORQ_ONLY_EXTRA_BODY_KEYS & set(params.get('extra_body') or {})):
        logger.debug('The Anthropic adapter does not send extra_body[{!r}]: only the Orq router reads it.', key)
    if params.get('fallbacks') is not None:
        extra_body['fallbacks'] = params['fallbacks']
    if extra_body:
        request['extra_body'] = extra_body


def usage_counts(message: Message) -> tuple[int, int, int, int]:
    """Fold Anthropic's split input counters into ``(prompt, completion, cache_read, cache_write)``; prompt includes both cache counters."""
    usage = message.usage
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    return usage.input_tokens + cache_read + cache_write, usage.output_tokens, cache_read, cache_write


def convert_usage(message: Message) -> CompletionUsage:
    """Build OpenAI ``CompletionUsage`` from an Anthropic message's usage."""
    prompt, completion, cache_read, cache_write = usage_counts(message)
    return CompletionUsage.model_validate({
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': prompt + completion,
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
    finish_reason = finish_reason_for(message.stop_reason)
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


def to_parsed_chat_completion(
    completion: ChatCompletion, *, response_format: Any, tools: Any = None
) -> ParsedChatCompletion[Any]:
    """Build a ``ParsedChatCompletion`` the way ``openai``'s ``chat.completions.parse`` does.

    Delegates to openai's own ``parse_chat_completion``, so the errors are openai's: a ``length`` finish raises ``LengthFinishReasonError`` and text that is not valid JSON for the Pydantic class raises ``pydantic.ValidationError``. A refusal is the exception: OpenAI reports a structured-output refusal as ``message.refusal`` with finish reason ``stop``, so that is what the ladder in ``common/structured_output.py`` reads, and a ``content_filter`` finish (which would raise) is not used for it.
    """
    choice = completion.choices[0]
    if choice.message.refusal and choice.finish_reason == 'content_filter':
        completion = completion.model_copy(deep=True)
        completion.choices[0].finish_reason = 'stop'
    return parse_chat_completion(
        response_format=response_format if response_format is not None else omit,
        input_tools=tools or omit,
        chat_completion=completion,
    )
