"""Import Orq trace conversations into evaluatorq's native datapoint shape."""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict, deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import httpx
from loguru import logger

from evaluatorq.common.messages import content_part_text
from evaluatorq.common.orq_client import orq_server_url
from evaluatorq.common.retry import with_retry
from evaluatorq.contracts import FunctionCall, Message, StrategyToolCall
from evaluatorq.openresponses.items import is_responses_item, parse_item
from evaluatorq.openresponses.otel_messages import items_to_input_messages
from evaluatorq.types import Trace, TraceInput

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_SPAN_FETCH_CONCURRENCY = 5
_API_PAGE_LIMIT = 200
_MESSAGE_FORMAT = Literal['chat_completions', 'responses', 'otel_genai']
_TRACE_MESSAGE_FORMAT = Literal['chat_completions', 'responses', 'otel_genai', 'mixed']
_ROLE = Literal['user', 'assistant', 'tool', 'system', 'developer']


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_orq_credentials(api_key: str | None, base_url: str | None) -> tuple[str, str]:
    key = api_key or os.environ.get('ORQ_API_KEY')
    if not key:
        raise ValueError('Missing Orq API key: set ORQ_API_KEY or pass api_key=.')
    # orq_server_url() is the single host resolver; re-deriving it here is how self-hosted deployments got ignored.
    host = (base_url or orq_server_url()).rstrip('/')
    return key, host


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped[:1] not in ('{', '[', '"'):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(',', ':'), sort_keys=True, default=str)


def _content_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        logger.warning('Unknown trace message content shape {}; JSON-encoding it.', type(value).__name__)
        return _json_text(value)
    text: list[str] = []
    for part in value:
        if isinstance(part, str):
            text.append(part)
            continue
        if not isinstance(part, dict):
            logger.warning('Unknown trace content part shape {}; JSON-encoding it.', type(part).__name__)
            text.append(_json_text(part))
            continue
        part_type = part.get('type')
        # Recorded traces carry untyped {'text': ...} parts, which the shared table does not accept.
        if (content := content_part_text(part if part_type is not None else {**part, 'type': 'text'})) is not None:
            if content:
                text.append(content)
            continue
        if part_type in {'image', 'image_url', 'input_image', 'input_file', 'file', 'audio', 'input_audio'}:
            logger.warning(
                'Trace message contains a {} part that cannot be scored as text; preserving a marker.', part_type
            )
            text.append(f'[{part_type}]')
            continue
        logger.warning('Unknown trace content part type {!r}; JSON-encoding it.', part_type)
        text.append(_json_text(part))
    return '\n'.join(text) if text else None


def _tool_call(raw: Any) -> StrategyToolCall | None:
    if not isinstance(raw, dict):
        logger.warning('Unknown trace tool-call shape {}; dropping it.', type(raw).__name__)
        return None
    function = _mapping(raw.get('function')) if isinstance(raw.get('function'), dict) else raw
    name = function.get('name')
    if not isinstance(name, str) or not name:
        logger.warning('Trace tool call has no function name; dropping it.')
        return None
    arguments = function.get('arguments', '{}')
    call_id = raw.get('call_id') or raw.get('id')
    if not isinstance(call_id, str) or not call_id:
        logger.warning('Trace tool call {!r} has no call ID; dropping it.', name)
        return None
    item_id = raw.get('id') if isinstance(raw.get('id'), str) and str(raw.get('id')).startswith('fc_') else None
    return StrategyToolCall(
        id=call_id,
        item_id=item_id,
        function=FunctionCall(name=name, arguments=_json_text(arguments)),
    )


def _optional_message_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    logger.warning('Trace message {} must be a string; dropping it.', field)
    return None


_MESSAGE_KEYS = frozenset({'role', 'content', 'tool_calls', 'tool_call_id', 'call_id', 'name', 'refusal'})


def _chat_message(raw: Any, *, default_role: _ROLE) -> Message | None:
    if isinstance(raw, str):
        return Message(role=default_role, content=raw)
    if not isinstance(raw, dict):
        logger.warning('Unknown trace message shape {}; dropping it.', type(raw).__name__)
        return None
    if not _MESSAGE_KEYS.intersection(raw):
        # A request-parameter mapping would otherwise parse as a user turn and stop the search before the real one.
        logger.warning('Trace message mapping carries no message keys (saw {}); dropping it.', sorted(raw)[:8])
        return None
    raw_role = raw.get('role') or default_role
    role = raw_role.strip().lower() if isinstance(raw_role, str) else raw_role
    if role not in {'user', 'assistant', 'tool', 'system', 'developer'}:
        logger.warning('Unknown trace message role {!r}; dropping it.', role)
        return None
    tool_calls = [call for value in raw.get('tool_calls') or [] if (call := _tool_call(value)) is not None]
    # A chat-completions refusal carries content=None and its text under 'refusal'.
    content = raw.get('content') if raw.get('content') is not None else raw.get('refusal')
    return Message(
        role=cast('_ROLE', role),
        content=_content_text(content),
        tool_calls=tool_calls or None,
        tool_call_id=_optional_message_string(raw.get('tool_call_id') or raw.get('call_id'), field='tool_call_id'),
        name=_optional_message_string(raw.get('name'), field='name'),
    )


def _chat_messages(value: Any, *, default_role: _ROLE) -> list[Message]:
    value = _decode_json(value)
    if isinstance(value, dict):
        if isinstance(value.get('messages'), list):
            value = value['messages']
        elif isinstance(value.get('choices'), list):
            value = [choice.get('message') for choice in value['choices'] if isinstance(choice, dict)]
        elif isinstance(value.get('message'), dict):
            value = [value['message']]
        elif isinstance(value.get('prompt'), str):
            value = [value['prompt']]
        elif isinstance(value.get('completion'), str):
            value = [value['completion']]
        else:
            value = [value]
    elif isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [message for raw in value if (message := _chat_message(raw, default_role=default_role)) is not None]


def _otel_parts(parts: list[Any]) -> tuple[list[str], list[StrategyToolCall], list[Message]]:
    """Normalize the content parts belonging to one OTel GenAI message."""
    text: list[str] = []
    tool_calls: list[StrategyToolCall] = []
    tool_responses: list[Message] = []
    media_types = {'image', 'image_url', 'input_image', 'input_file', 'file', 'audio', 'input_audio'}
    for part in parts:
        if not isinstance(part, dict):
            logger.warning('Unknown OTel GenAI part shape {}; dropping it.', type(part).__name__)
            continue
        part_type = part.get('type')
        if part_type in {None, 'text', 'refusal'}:
            content = part.get('content') or part.get('text') or part.get('refusal')
            if isinstance(content, str):
                text.append(content)
        elif part_type == 'tool_call':
            if (call := _tool_call(part)) is not None:
                tool_calls.append(call)
        elif part_type == 'tool_call_response':
            call_id = part.get('id') or part.get('call_id')
            if isinstance(call_id, str) and call_id:
                tool_responses.append(
                    Message(
                        role='tool',
                        tool_call_id=call_id,
                        name=part.get('name'),
                        content=_json_text(part.get('response', part.get('result', part.get('output', '')))),
                    )
                )
            else:
                logger.warning('OTel GenAI tool response has no call ID; dropping it.')
        elif part_type in media_types:
            logger.warning(
                'Trace message contains a {} part that cannot be scored as text; preserving a marker.', part_type
            )
            text.append(f'[{part_type}]')
        elif part_type == 'reasoning':
            logger.debug('Dropping an OTel GenAI reasoning part from the imported transcript.')
        else:
            logger.warning('Unknown OTel GenAI part type {!r}; preserving JSON text.', part_type)
            text.append(_json_text(part))
    return text, tool_calls, tool_responses


def _otel_messages(value: Any, *, default_role: _ROLE) -> list[Message]:
    value = _decode_json(value)
    if isinstance(value, dict) and isinstance(value.get('messages'), list):
        value = value['messages']
    if isinstance(value, dict) and isinstance(value.get('parts'), list):
        # One bare OTel message object: _chat_messages reads only 'content' and would drop its parts.
        value = [value]
    if not isinstance(value, list):
        return _chat_messages(value, default_role=default_role)
    messages: list[Message] = []
    for raw in value:
        if not isinstance(raw, dict) or not isinstance(raw.get('parts'), list):
            message = _chat_message(raw, default_role=default_role)
            if message is not None:
                messages.append(message)
            continue
        raw_role = raw.get('role') or default_role
        role = raw_role.strip().lower() if isinstance(raw_role, str) else raw_role
        text, tool_calls, tool_responses = _otel_parts(raw['parts'])
        if role == 'tool' and tool_responses:
            messages.extend(tool_responses)
        elif role in {'user', 'assistant', 'tool', 'system', 'developer'} and (text or tool_calls):
            messages.append(
                Message(
                    role=cast('_ROLE', role),
                    content='\n'.join(text) if text else None,
                    tool_calls=tool_calls or None,
                )
            )
            messages.extend(tool_responses)
        elif role not in {'user', 'assistant', 'tool', 'system', 'developer'}:
            logger.warning('Unknown OTel GenAI message role {!r}; dropping it.', role)
    return messages


def _responses_messages(value: Any, *, default_role: _ROLE) -> list[Message]:
    value = _decode_json(value)
    if isinstance(value, dict) and not is_responses_item(value) and ('input' in value or 'output' in value):
        value = value.get('input') if default_role == 'user' else value.get('output')
    if isinstance(value, str):
        return [Message(role=default_role, content=value)]
    if isinstance(value, dict):
        if is_responses_item(value):
            value = [value]
        elif isinstance(value.get('messages'), list):
            return _chat_messages(value, default_role=default_role)
        elif 'role' in value or 'content' in value:
            return _chat_messages([value], default_role=default_role)
    # OTel parts carry no item id, so the ``fc_`` id of each call is recovered from the items themselves.
    parsed = [parse_item(item) for item in value if is_responses_item(item)] if isinstance(value, list) else []
    item_ids = {
        item.call_id: item.item_id
        for item in parsed
        if item.kind == 'tool_call' and item.call_id and item.item_id and item.item_id.startswith('fc_')
    }
    # The input renderer for both sides: the output renderer drops tool results, which an imported transcript needs.
    otel = items_to_input_messages(value, default_role=default_role)
    messages = _otel_messages(otel, default_role=default_role)
    for message in messages:
        for call in message.tool_calls or []:
            call.item_id = item_ids.get(call.id)
    return messages


def _looks_like_otel(value: Any) -> bool:
    decoded = _decode_json(value)
    if isinstance(decoded, dict):
        decoded = decoded.get('messages', [decoded])
    return isinstance(decoded, list) and any(isinstance(item, dict) and 'parts' in item for item in decoded)


def _looks_like_responses(value: Any) -> bool:
    decoded = _decode_json(value)
    if isinstance(decoded, dict):
        # Either side of the Responses envelope may be a scalar or a Chat-style shorthand list.
        if 'input' in decoded or 'output' in decoded:
            return True
        return is_responses_item(decoded)
    return isinstance(decoded, list) and any(isinstance(item, dict) and is_responses_item(item) for item in decoded)


def _parse_messages(
    value: Any, *, hinted: _MESSAGE_FORMAT | None = None, default_role: _ROLE
) -> tuple[list[Message], _MESSAGE_FORMAT]:
    detected: _MESSAGE_FORMAT
    if hinted is not None:
        detected = hinted
    elif _looks_like_otel(value):
        detected = 'otel_genai'
    elif _looks_like_responses(value):
        detected = 'responses'
    else:
        detected = 'chat_completions'
    if detected == 'otel_genai':
        return _otel_messages(value, default_role=default_role), detected
    if detected == 'responses':
        return _responses_messages(value, default_role=default_role), detected
    return _chat_messages(value, default_role=default_role), detected


def _nested(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _attribute_side_candidates(attributes: dict[str, Any], side: str) -> list[tuple[_MESSAGE_FORMAT | None, Any]]:
    gen_ai = _mapping(attributes.get('gen_ai'))
    openresponses = _mapping(attributes.get('openresponses'))
    orq = _mapping(attributes.get('orq'))
    orq_openresponses = _mapping(orq.get('openresponses'))
    openresponses_key = f'openresponses.{side}'
    candidates: list[tuple[_MESSAGE_FORMAT | None, Any]] = [
        ('otel_genai', attributes.get(f'gen_ai.{side}.messages')),
        ('otel_genai', attributes.get(f'gen_ai.{side}')),
        (
            'otel_genai',
            attributes.get('gen_ai.input.prompt' if side == 'input' else 'gen_ai.output.completion'),
        ),
        ('otel_genai', _nested(gen_ai, side, 'messages')),
        ('responses', attributes.get(openresponses_key)),
        ('responses', openresponses.get(side)),
    ]
    full_request = _decode_json(
        attributes.get('openresponses.request')
        or openresponses.get('request')
        or attributes.get('orq.openresponses.request')
        or orq_openresponses.get('request')
    )
    full_response = _decode_json(
        attributes.get('openresponses.response')
        or openresponses.get('response')
        or attributes.get('orq.openresponses.response')
        or orq_openresponses.get('response')
    )
    if side == 'input':
        candidates.append(('responses', _nested(full_request, 'input') if isinstance(full_request, dict) else None))
    else:
        candidates.append(('responses', _nested(full_response, 'output') if isinstance(full_response, dict) else None))
    candidates.extend([
        (None, gen_ai.get(side)),
        (None, attributes.get(side)),
    ])
    return [(hint, value) for hint, value in candidates if value is not None]


def _message_candidates(span: dict[str, Any], side: str) -> list[tuple[_MESSAGE_FORMAT | None, Any]]:
    """Return message payloads in the documented span-location precedence order."""
    candidates = _attribute_side_candidates(_mapping(span.get('attributes')), side)
    events = span.get('events')
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                candidates.extend(_attribute_side_candidates(_mapping(event.get('attributes')), side))
    if span.get(side) is not None:
        candidates.append((None, span[side]))
    return candidates


def _parse_exchange(
    span: dict[str, Any],
) -> tuple[list[Message], list[Message], _TRACE_MESSAGE_FORMAT | None]:
    sides: list[tuple[list[Message], _MESSAGE_FORMAT]] = []
    side_roles: tuple[tuple[str, _ROLE], ...] = (('input', 'user'), ('output', 'assistant'))
    for side, default_role in side_roles:
        parsed: tuple[list[Message], _MESSAGE_FORMAT] | None = None
        for hinted, value in _message_candidates(span, side):
            messages, detected = _parse_messages(value, hinted=hinted, default_role=default_role)
            # Content-less messages would stop the search at a shape that merely looked like a conversation.
            if any(message.content or message.tool_calls or message.tool_call_id for message in messages):
                parsed = messages, detected
                break
        if parsed is not None:
            sides.append(parsed)
        else:
            sides.append(([], 'chat_completions'))
    input_messages, _input_format = sides[0]
    output_messages, _output_format = sides[1]
    formats = {fmt for found, fmt in sides if found}
    message_format = cast(
        '_TRACE_MESSAGE_FORMAT | None', next(iter(formats)) if len(formats) == 1 else ('mixed' if formats else None)
    )
    return input_messages, output_messages, message_format


def _span_id(span: dict[str, Any], index: int) -> str:
    value = span.get('span_id') or span.get('_id') or span.get('id')
    return str(value) if value else f'__span_{index}'


def _parent_id(span: dict[str, Any]) -> str | None:
    value = span.get('parent_span_id') or span.get('parent_id')
    return str(value) if value else None


_EVALUATOR_ATTRIBUTE_PREFIXES = ('orq.evaluator.', 'gen_ai.evaluation.')
"""Attribute namespaces that mark a span as evaluator output.

The trailing dot is load-bearing. A bare ``orq.evaluator`` prefix also matches
``orq.evaluatorq_run_id``, the run-level attribute evaluatorq stamps on the root
span of every run it traces — which made every evaluatorq-produced trace
unimportable, reported as "every span belongs to an evaluator span or its subtree".
"""


def _is_evaluator(span: dict[str, Any]) -> bool:
    span_type = str(span.get('type') or '').lower()
    if 'evaluator' in span_type or 'evaluation' in span_type:
        return True
    attributes = _mapping(span.get('attributes'))
    orq_span_type = str(attributes.get('orq.span_type') or _nested(attributes, 'orq', 'span_type')).lower()
    if 'evaluator' in orq_span_type or 'evaluation' in orq_span_type:
        return True
    return any(
        str(key) in {'orq.evaluator', 'gen_ai.evaluation'} or str(key).startswith(_EVALUATOR_ATTRIBUTE_PREFIXES)
        for key in attributes
    )


def _time_key(span: dict[str, Any], index: int) -> tuple[float, int]:
    value = span.get('started_at') or span.get('start_time') or span.get('ended_at') or span.get('end_time')
    if isinstance(value, (int, float)):
        return float(value), index
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp(), index
        except ValueError:
            logger.warning('Could not parse trace span timestamp {!r}; using response order.', value)
    return float(index), index


def _message_text(message: Message) -> str:
    return message.content if isinstance(message.content, str) else ''


def _first_attr(spans: list[dict[str, Any]], *names: str) -> str | None:
    for span in spans:
        attributes = _mapping(span.get('attributes'))
        for name in names:
            value = attributes.get(name)
            if value is None and '.' in name:
                value = _nested(attributes, *name.split('.'))
            if isinstance(value, str) and value:
                return value
    return None


def _retrievals(span: dict[str, Any]) -> list[str]:
    attributes = _mapping(span.get('attributes'))
    value = attributes.get('retrievals') or attributes.get('gen_ai.retrievals') or attributes.get('orq.retrievals')
    value = _decode_json(value)
    if value is None:
        return []
    if not isinstance(value, list):
        logger.warning('Unknown trace retrievals shape {}; preserving it as one JSON string.', type(value).__name__)
        value = [value]
    return [_json_text(item) for item in value]


def _human_expected_output(trace_metadata: dict[str, Any]) -> str | None:
    for field in ('feedback', 'annotations'):
        annotations = trace_metadata.get(field)
        if isinstance(annotations, dict):
            annotations = [annotations]
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            property_name = annotation.get('property') or annotation.get('name') or annotation.get('key')
            value = annotation.get('value')
            if (
                isinstance(property_name, str)
                and property_name.strip().lower() == 'correction'
                and isinstance(value, str)
                and value.strip()
            ):
                return value
    return None


def _failed_trace(
    trace_id: str,
    error: str,
    *,
    requested_span_id: str | None = None,
) -> Trace:
    logger.warning('Could not import trace {}: {}', trace_id, error)
    return Trace(trace_id=trace_id, requested_span_id=requested_span_id, import_error=error)


def _evaluator_subtree_ids(indexed: list[tuple[str, dict[str, Any], int]]) -> set[str]:
    """Return evaluator span IDs together with every descendant span ID."""
    children: dict[str | None, list[str]] = defaultdict(list)
    for span_id, span, _ in indexed:
        children[_parent_id(span)].append(span_id)
    evaluator_ids = {span_id for span_id, span, _ in indexed if _is_evaluator(span)}
    queue = deque(evaluator_ids)
    while queue:
        for child_id in children.get(queue.popleft(), ()):
            if child_id not in evaluator_ids:
                evaluator_ids.add(child_id)
                queue.append(child_id)
    return evaluator_ids


def _trace_from_spans(
    trace_id: str,
    spans: list[dict[str, Any]],
    *,
    requested_span_id: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> Trace:
    """Select and normalize one exchange from a trace's spans.

    An explicit span request is exact and is never replaced by a parent span.
    Trace-only and query mode select the latest eligible span and walk toward
    its parent only when that selected span has no messages.
    """
    if not spans:
        return _failed_trace(trace_id, 'the trace has no spans.', requested_span_id=requested_span_id)
    indexed = [(_span_id(span, index), span, index) for index, span in enumerate(spans)]
    by_id = {span_id: span for span_id, span, _ in indexed}
    evaluator_ids = _evaluator_subtree_ids(indexed)
    if requested_span_id is not None:
        requested_span = by_id.get(requested_span_id)
        if requested_span is None:
            return _failed_trace(
                trace_id,
                f'requested span {requested_span_id!r} was not found.',
                requested_span_id=requested_span_id,
            )
        if requested_span_id in evaluator_ids:
            return _failed_trace(
                trace_id,
                f'requested span {requested_span_id!r} belongs to an evaluator span or its subtree.',
                requested_span_id=requested_span_id,
            )
        selected_id = requested_span_id
        selected_span = requested_span
        input_messages, output_messages, message_format = _parse_exchange(selected_span)
        if not input_messages and not output_messages:
            return _failed_trace(
                trace_id,
                f'requested span {requested_span_id!r} has no input or output messages.',
                requested_span_id=requested_span_id,
            )
    else:
        eligible = [(span_id, span, index) for span_id, span, index in indexed if span_id not in evaluator_ids]
        if not eligible:
            return _failed_trace(trace_id, 'every span belongs to an evaluator span or its subtree.')
        leaf_id, _leaf_span, _leaf_index = max(eligible, key=lambda item: _time_key(item[1], item[2]))
        current_id: str | None = leaf_id
        selected: tuple[str, dict[str, Any], list[Message], list[Message], _TRACE_MESSAGE_FORMAT | None] | None = None
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            span = by_id.get(current_id)
            if span is None:
                break
            candidate_input, candidate_output, candidate_format = _parse_exchange(span)
            if candidate_input or candidate_output:
                selected = current_id, span, candidate_input, candidate_output, candidate_format
                break
            current_id = _parent_id(span)
        if selected is not None:
            selected_id, selected_span, input_messages, output_messages, message_format = selected
        else:
            selected_id = None
            selected_span = None
            input_messages = []
            output_messages = []
            message_format = None
    if selected_span is None:
        return _failed_trace(
            trace_id,
            'no non-evaluator span on the latest span path contains input or output messages.',
        )
    message_span_id = selected_id
    message_span = selected_span
    tools_called = [
        call.function.name
        for message in [*input_messages, *output_messages]
        for call in (message.tool_calls or [])
        if call.function.name
    ]
    tools_called = list(dict.fromkeys(tools_called))
    query = next(
        (
            _message_text(message)
            for message in reversed([*input_messages, *output_messages])
            if message.role == 'user' and _message_text(message).strip()
        ),
        None,
    )
    metadata_source = trace_metadata or {}
    metadata = {
        key: value
        for key, value in {
            'trace_name': metadata_source.get('name'),
            'span_name': message_span.get('name'),
            'span_type': message_span.get('type'),
            'operation': message_span.get('operation'),
            'provider': message_span.get('provider'),
            'model': message_span.get('model'),
            'status': message_span.get('status'),
        }.items()
        if value is not None
    }
    custom_metadata = metadata_source.get('metadata')
    if isinstance(custom_metadata, dict):
        metadata.update(custom_metadata)
    return Trace(
        trace_id=trace_id,
        requested_span_id=requested_span_id,
        message_span_id=None if message_span_id is None or message_span_id.startswith('__span_') else message_span_id,
        message_format=message_format,
        input_messages=input_messages,
        output_messages=output_messages,
        query=query,
        retrievals=_retrievals(message_span),
        tools_called=tools_called,
        session_id=_first_attr(spans, 'session_id', 'session.id', 'gen_ai.conversation.id'),
        actor_id=_first_attr(spans, 'actor_id', 'actor.id', 'enduser.id'),
        thread_id=_first_attr(spans, 'thread_id', 'thread.id', 'gen_ai.thread.id'),
        metadata=metadata,
        expected_output=_human_expected_output(metadata_source),
    )


def _datetime_ms(value: datetime | None) -> int | None:
    """Epoch milliseconds for an Orq query bound.

    ``TraceInput`` normalizes naive datetimes to UTC, so this conversion no
    longer depends on the host's local timezone — the same query used to shift
    by the offset of whichever machine ran it.
    """
    return int(value.timestamp() * 1000) if value is not None else None


async def _list_trace_rows(
    client: httpx.AsyncClient,
    source: TraceInput,
    *,
    host: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Page through the trace-list endpoint until the query limit is reached.

    A page that is not a JSON object, or that carries no usable trace ID, stops
    pagination with a warning rather than raising: the caller still gets the rows
    it already has, and the reason is in the log.
    """
    rows: list[dict[str, Any]] = []
    page = 1
    query_limit = source.query_limit
    while len(rows) < query_limit:
        before = len(rows)
        body: dict[str, Any] = {
            'filters': {'operator': 'and', 'filters': source.filters, 'search': source.search},
            'limit': min(query_limit - len(rows), _API_PAGE_LIMIT),
            'page': page,
            'fields': [],
        }
        if (start_date := _datetime_ms(source.start_time)) is not None:
            body['start_date'] = start_date
        if (end_date := _datetime_ms(source.end_time)) is not None:
            body['end_date'] = end_date
        try:

            async def list_page(body: dict[str, Any] = body) -> Any:
                response = await client.post(f'{host}/v2/traces/v3oql', headers=headers, json=body)
                response.raise_for_status()
                return response.json()

            payload = await with_retry(list_page, label='Orq trace list')
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'Failed to list Orq traces: {exc}') from exc
        if not isinstance(payload, dict):
            logger.warning('Orq trace list returned a {}, not an object; stopping pagination.', type(payload).__name__)
            break
        data = payload.get('data', [])
        if not isinstance(data, list):
            logger.warning(
                'Orq trace list returned a {} data field, not a list; stopping pagination.', type(data).__name__
            )
            break
        rows.extend(row for row in data if isinstance(row, dict) and row.get('trace_id'))
        if not data or not payload.get('has_more'):
            break
        if len(rows) == before:
            logger.warning('Stopped paginating traces: a page contained no usable trace IDs.')
            break
        page += 1
    return rows


async def fetch_traces(
    source: TraceInput,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[Trace]:
    """Fetch Orq traces and normalize each one to a canonical trace.

    One unreadable trace becomes a ``Trace`` with ``import_error`` set;
    other traces in the batch remain available. No trace is silently discarded.
    Partition the result with `partition_traces` rather than filtering by hand —
    every surface is meant to report import failures the same way.

    The HTTP calls retry through `common.retry.with_retry` — including the 429s
    and 5xx that ``raise_for_status`` raises as ``httpx.HTTPStatusError`` — and the
    shared ``httpx.AsyncClient`` adds no retry layer of its own. Raw httpx rather
    than the Orq SDK because ``traces.search`` requires a ``from_``/``to`` window
    that ``TraceInput`` does not, and its span models would need a second parser.
    """
    key, host = _resolve_orq_credentials(api_key, base_url)
    headers = {'Authorization': f'Bearer {key}'}
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=60.0)
    try:
        rows = (
            [{'trace_id': source.trace_id}]
            if source.trace_id is not None
            else await _list_trace_rows(client, source, host=host, headers=headers)
        )

        semaphore = asyncio.Semaphore(_SPAN_FETCH_CONCURRENCY)

        async def fetch_one(row: dict[str, Any]) -> Trace:
            trace_id = str(row['trace_id'])
            async with semaphore:

                async def fetch_spans(trace_id: str = trace_id) -> Any:
                    response = await client.get(f'{host}/v2/traces/{trace_id}/v3spans', headers=headers)
                    response.raise_for_status()
                    return response.json()

                try:
                    spans = await with_retry(fetch_spans, label=f'Orq trace {trace_id} spans')
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    return _failed_trace(trace_id, f'could not fetch its spans: {exc}')
            if not isinstance(spans, list):
                return _failed_trace(trace_id, f'its spans payload is {type(spans).__name__}, not a list.')
            if not all(isinstance(span, dict) for span in spans):
                # One non-object element would raise inside asyncio.gather and take the whole batch with it.
                return _failed_trace(trace_id, 'its spans payload contains a non-object span.')
            return _trace_from_spans(
                trace_id,
                spans,
                requested_span_id=source.span_id,
                trace_metadata=row,
            )

        selected_rows = rows if source.trace_id is not None else rows[: source.query_limit]
        if not selected_rows and source.trace_id is None:
            # A green run over no datapoints reads as "everything passed".
            logger.warning(
                'Trace query matched no traces (limit={}, search={!r}, filters={}, start_time={}, end_time={}).',
                source.query_limit,
                source.search,
                len(source.filters),
                source.start_time,
                source.end_time,
            )
        return await asyncio.gather(*(fetch_one(row) for row in selected_rows if row.get('trace_id')))
    finally:
        if owned:
            await client.aclose()


def partition_traces(traces: Sequence[Trace], *, caller: str) -> tuple[list[Trace], list[Trace]]:
    """Split imported traces into usable and failed, logging the failures once.

    Every surface that consumes `fetch_traces` reports import failures through
    this one function. Before it, red team raised on the first failure and lost
    the whole batch, simulation dropped them with its own message, and core
    evaluation turned each into a row that raised — three answers to one event,
    chosen by which entry point the caller happened to use.
    """
    usable = [trace for trace in traces if trace.import_error is None]
    failed = [trace for trace in traces if trace.import_error is not None]
    if failed:
        details = '; '.join(f'{trace.trace_id}: {trace.import_error}' for trace in failed)
        logger.warning(
            '{}: {} of {} imported trace(s) failed and were excluded: {}', caller, len(failed), len(traces), details
        )
    return usable, failed


class TraceBatch(NamedTuple):
    """A resolved trace source: every trace in order, and its usable/failed split."""

    traces: list[Trace]
    usable: list[Trace]
    failed: list[Trace]


async def load_traces(
    source: TraceInput | Iterable[Trace],
    *,
    caller: str,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> TraceBatch:
    """Resolve a trace source for any surface: fetch a `TraceInput`, take loaded traces as-is.

    Import failures are logged once, through `partition_traces`. What an empty or
    all-failed batch means is the surface's own rule, so this never raises on one.
    """
    if isinstance(source, TraceInput):
        traces = await fetch_traces(source, api_key=api_key, base_url=base_url, http_client=http_client)
    else:
        traces = list(source)
    usable, failed = partition_traces(traces, caller=caller)
    return TraceBatch(traces, usable, failed)


__all__ = ['TraceBatch', 'fetch_traces', 'load_traces', 'partition_traces']
