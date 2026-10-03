"""OTel GenAI semantic-convention data models (plain data, no opentelemetry import).

`OtelTrace.from_orq` reads raw Orq span dicts: flat or nested attributes, OTel `parts` messages, and the
shapes Orq actually exports (index-keyed dicts, chat-completions messages, span-summary usage and cost).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from evaluatorq.common.trace_input import parent_span_id_of, span_id_of
from evaluatorq.formats._shared import parse_iso

if TYPE_CHECKING:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.formats.chat import ChatConversation
    from evaluatorq.formats.responses import ResponsesConversation

_OPEN = ConfigDict(extra='allow', frozen=True)
_ORQ_PARTS = frozenset({'refusal', 'data'})  # emitted by openresponses.otel_messages, kept generic without a warning


class OtelTextPart(BaseModel):
    model_config = _OPEN
    type: Literal['text']
    content: str


class OtelReasoningPart(BaseModel):
    model_config = _OPEN
    type: Literal['reasoning']
    content: str


class OtelToolCallPart(BaseModel):
    model_config = _OPEN
    type: Literal['tool_call']
    id: str | None = None
    name: str
    arguments: Any = None


class OtelToolCallResponsePart(BaseModel):
    model_config = _OPEN
    type: Literal['tool_call_response']
    id: str | None = None
    response: Any


class OtelServerToolCallPart(BaseModel):
    model_config = _OPEN
    type: Literal['server_tool_call']
    id: str | None = None
    name: str
    server_tool_call: Any


class OtelServerToolCallResponsePart(BaseModel):
    model_config = _OPEN
    type: Literal['server_tool_call_response']
    id: str | None = None
    server_tool_call_response: Any


class OtelBlobPart(BaseModel):
    model_config = _OPEN
    type: Literal['blob']
    mime_type: str | None = None
    modality: str
    content: str


class OtelFilePart(BaseModel):
    model_config = _OPEN
    type: Literal['file']
    mime_type: str | None = None
    modality: str
    file_id: str


class OtelUriPart(BaseModel):
    model_config = _OPEN
    type: Literal['uri']
    mime_type: str | None = None
    modality: str
    uri: str


class OtelCompactionPart(BaseModel):
    model_config = _OPEN
    type: Literal['compaction']
    id: str | None = None
    content: Any = None


class OtelGenericPart(BaseModel):
    model_config = _OPEN
    type: str


OtelPart = Union[  # noqa: UP007  (pydantic needs a runtime union on 3.10)
    OtelTextPart,
    OtelReasoningPart,
    OtelToolCallPart,
    OtelToolCallResponsePart,
    OtelServerToolCallPart,
    OtelServerToolCallResponsePart,
    OtelBlobPart,
    OtelFilePart,
    OtelUriPart,
    OtelCompactionPart,
    OtelGenericPart,
]
_PART_MODELS: dict[str, type[BaseModel]] = {  # keyed by the `type` literal
    'text': OtelTextPart,
    'reasoning': OtelReasoningPart,
    'tool_call': OtelToolCallPart,
    'tool_call_response': OtelToolCallResponsePart,
    'server_tool_call': OtelServerToolCallPart,
    'server_tool_call_response': OtelServerToolCallResponsePart,
    'blob': OtelBlobPart,
    'file': OtelFilePart,
    'uri': OtelUriPart,
    'compaction': OtelCompactionPart,
}
_PART_CLASSES = (*_PART_MODELS.values(), OtelGenericPart)


def parse_part(raw: Any) -> OtelPart:
    """Parse one part dict. Unknown `type` becomes OtelGenericPart (warning, except Orq refusal/data).

    A `type` that is not a string is kept as its `str()`, and a part that is not a dict as `{"type": "unknown",
    "value": part}`, so the generic fallback never raises.
    """
    if isinstance(raw, _PART_CLASSES):
        return cast('OtelPart', raw)
    if isinstance(raw, dict) and 'type' not in raw and 'kind' in raw:
        raw = _a2a_part(cast('dict[str, Any]', raw))
    kind = raw.get('type') if isinstance(raw, dict) else None
    model = _PART_MODELS.get(kind) if isinstance(kind, str) else None
    if model is not None:
        return cast('OtelPart', model.model_validate(raw))
    if not (isinstance(kind, str) and kind in _ORQ_PARTS):
        logger.warning('Unknown OTel message part type {!r}; keeping it as a generic part', kind)
    if not isinstance(raw, dict):
        return OtelGenericPart.model_validate({'type': 'unknown', 'value': raw})
    generic = {**cast('dict[str, Any]', raw), 'type': kind if isinstance(kind, str) else str(kind)}
    return OtelGenericPart.model_validate(generic)


def _a2a_part(raw: dict[str, Any]) -> dict[str, Any]:
    """Read an A2A-style `{kind, text}` part (Orq's root `gen_ai.input.message`) as an OTel part."""
    part = {key: value for key, value in raw.items() if key != 'kind'}
    part['type'] = raw['kind']
    if raw['kind'] == 'text' and 'content' not in part and 'text' in part:
        part['content'] = part.pop('text')
    return part


def _as_list(value: Any) -> Any:
    """Turn an Orq index-keyed dict (`{"0": ..., "1": ...}`) into a list ordered by int key; else return as-is."""
    if isinstance(value, dict) and value and all(isinstance(key, str) and key.isdigit() for key in value):
        items = cast('dict[str, Any]', value)
        return [items[key] for key in sorted(items, key=int)]
    return value


class OtelMessage(BaseModel):
    """One GenAI message. `finish_reason` is read when present; converters write it as a span attribute.

    Orq's `agent` role (A2A naming) is read as `assistant`.
    """

    model_config = _OPEN
    role: str
    parts: list[OtelPart]
    name: str | None = None
    finish_reason: str | None = None

    @field_validator('role')
    @classmethod
    def _assistant_role(cls, value: str) -> str:
        return 'assistant' if value == 'agent' else value

    @field_validator('parts', mode='before')
    @classmethod
    def _parse_parts(cls, value: Any) -> Any:
        value = _as_list(value)
        return [parse_part(part) for part in value] if isinstance(value, list) else value


class OtelUsage(BaseModel):
    """Token usage and cost of one span. Unreported values stay None, never 0."""

    model_config = ConfigDict(frozen=True)
    input_tokens: int | float | None = None
    output_tokens: int | float | None = None
    cached_tokens: int | float | None = None
    reasoning_tokens: int | float | None = None
    cost_usd: int | float | None = None


class OtelSpan(BaseModel):
    """One span. `span_type` keeps Orq's raw wrapper type; parsed message attributes leave `attributes`.

    A naive `start_time` or `end_time` is read as UTC.
    """

    model_config = ConfigDict(frozen=True)
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str = ''
    span_type: str | None = None
    operation: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: Literal['ok', 'error', 'unset'] = 'unset'
    error_type: str | None = None
    input_messages: list[OtelMessage] | None = None
    output_messages: list[OtelMessage] | None = None
    system_instructions: list[OtelPart] | None = None
    attributes: dict[str, Any] = {}

    @field_validator('start_time', 'end_time')
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        return parse_time(value)


class OtelTrace(BaseModel):
    """A list of spans forming one trace (a forest when parents are missing).

    Every span that names a trace id must name the same one; spans with an empty trace id are accepted.

    Raises:
        ValueError: The spans carry more than one distinct trace id.
    """

    model_config = ConfigDict(frozen=True)
    spans: list[OtelSpan]

    @model_validator(mode='after')
    def _one_trace_id(self) -> OtelTrace:
        trace_ids = sorted({span.trace_id for span in self.spans if span.trace_id})
        if len(trace_ids) > 1:
            msg = f'OtelTrace spans carry {len(trace_ids)} trace ids {trace_ids}; split them into one OtelTrace per trace id'
            raise ValueError(msg)
        return self

    @classmethod
    def from_orq(cls, spans: list[dict[str, Any]]) -> OtelTrace:
        """Parse raw Orq span dicts (the full tree, messages kept typed and lossless)."""
        return cls(spans=[_span_from_orq(raw, index) for index, raw in enumerate(spans)])

    def roots(self) -> list[OtelSpan]:
        """Spans with no parent, or whose parent is not in this trace, in start-time order."""
        ids = {span.span_id for span in self.spans}
        return order_spans([s for s in self.spans if s.parent_span_id is None or s.parent_span_id not in ids])

    def children(self, span_id: str) -> list[OtelSpan]:
        """Direct children of `span_id` in start-time order (no time last, ties by list order)."""
        return order_spans([s for s in self.spans if s.parent_span_id == span_id])

    def to_atif(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> AtifTrajectory:
        """Convert to an ATIF trajectory (subagent `invoke_agent` subtrees become subagent trajectories).

        Lost: see `convert_otel_atif.otel_to_atif` for the mapping. Extra output choices (warned), span attributes
        other than model, usage, finish reasons and error type, and spans that are not `chat`, `execute_tool` or
        `invoke_agent`.
        """
        from evaluatorq.formats import convert_otel_atif

        return convert_otel_atif.otel_to_atif(self, agent_name=agent_name, agent_version=agent_version)

    def to_responses(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> ResponsesConversation:
        """Convert to a Responses transcript through ATIF.

        Lost: everything lost by `OtelTrace.to_atif` and `AtifTrajectory.to_responses` (span times and ancestry,
        subagent trees, per-call metrics beyond one `Response` per agent step).
        """
        return self.to_atif(agent_name=agent_name, agent_version=agent_version).to_responses()

    def to_chat(self, *, agent_name: str = 'unknown', agent_version: str = 'unknown') -> ChatConversation:
        """Convert to chat through ATIF and Responses.

        Lost: everything lost by `to_atif` and `ResponsesConversation.to_chat` (reasoning, span times and
        ancestry, per-call metrics, subagent trees, media).
        """
        return self.to_atif(agent_name=agent_name, agent_version=agent_version).to_responses().to_chat()


def order_spans(spans: list[OtelSpan], *, fallback_end: bool = False) -> list[OtelSpan]:
    """Spans in start-time order; with `fallback_end`, a span with no start sorts by its end.

    A span with no usable time goes last, and ties keep list order.
    """

    def key(span: OtelSpan) -> tuple[bool, float]:
        at = span.start_time or (span.end_time if fallback_end else None)
        return (True, 0.0) if at is None else (False, at.timestamp())

    return sorted(spans, key=key)


# --- raw Orq span parsing ---------------------------------------------------------------------------

_MESSAGE_ATTRIBUTES = (
    ('input_messages', 'gen_ai.input.messages'),
    ('input_messages', 'gen_ai.input.message'),  # Orq root span: one message, singular key
    ('input_messages', 'gen_ai.input'),  # Orq router root: direct message list
    ('output_messages', 'gen_ai.output.messages'),
    ('output_messages', 'gen_ai.output'),  # Orq router root: direct message list
)
_SYSTEM_INSTRUCTIONS = 'gen_ai.system_instructions'
# Structured values kept whole by `flatten_attributes` (their inner dicts are data, not attribute namespaces).
_LEAF_ATTRIBUTES = frozenset({
    *(key for _, key in _MESSAGE_ATTRIBUTES if key not in ('gen_ai.input', 'gen_ai.output')),
    _SYSTEM_INSTRUCTIONS,
    'gen_ai.tool.definitions',
})
_OPERATION_ALIASES = {'chat-completion': 'chat'}
_SPAN_TYPE_OPERATIONS = {'span.chat_completion': 'chat', 'span.agent_execution': 'invoke_agent'}
# span_usage field -> attribute keys in priority order; the first key is where Orq's span summary is copied.
_USAGE_KEYS: dict[str, tuple[str, ...]] = {
    'input_tokens': ('gen_ai.usage.input_tokens', 'gen_ai.usage.prompt_tokens'),
    'output_tokens': ('gen_ai.usage.output_tokens', 'gen_ai.usage.completion_tokens'),
    'cached_tokens': ('gen_ai.usage.cache_read.input_tokens',),
    'reasoning_tokens': ('gen_ai.usage.reasoning.output_tokens',),
    'cost_usd': ('gen_ai.usage.total_cost', 'gen_ai.usage.cost'),
}
_SUMMARY_USAGE = {
    'input_tokens': 'prompt_tokens',
    'output_tokens': 'completion_tokens',
    'cached_tokens': 'prompt_cached_tokens',
    'reasoning_tokens': 'completion_reasoning_tokens',
}


def parse_time(value: object) -> datetime | None:
    """Parse an ISO string, epoch seconds or epoch milliseconds to an aware datetime (naive means UTC)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 10_000_000_000 else value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            logger.warning('Span time {!r} is out of range ({}); leaving it unset', value, exc)
            return None
    if isinstance(value, str):
        try:
            parsed = parse_iso(value)
        except ValueError:
            pass
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    logger.warning('Unparseable span time {!r}', value)
    return None


def flatten_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested attribute dicts to dotted keys. A caller-supplied flat key wins over a nested duplicate."""
    derived: dict[str, Any] = {}
    direct: dict[str, Any] = {}

    def is_leaf(key: str, value: dict[str, Any]) -> bool:
        if key in _LEAF_ATTRIBUTES:
            return True
        return key in ('gen_ai.input', 'gen_ai.output') and (
            'role' in value
            or isinstance(value.get('message'), dict)
            or isinstance(value.get('messages'), list)
            or (bool(value) and all(index.isdigit() for index in value))
        )

    def walk(prefix: str, value: dict[str, Any]) -> None:
        for key, item in value.items():
            dotted = f'{prefix}.{key}'
            if isinstance(item, dict) and not is_leaf(dotted, cast('dict[str, Any]', item)):
                walk(dotted, cast('dict[str, Any]', item))
            else:
                derived.setdefault(dotted, item)

    for key, value in attrs.items():
        if isinstance(value, dict) and not is_leaf(key, cast('dict[str, Any]', value)):
            walk(key, cast('dict[str, Any]', value))
        else:
            direct[key] = value
    return {**derived, **direct}


def span_usage(span: OtelSpan) -> OtelUsage:
    """Read usage and cost from `gen_ai.usage.*` (semconv names first, legacy prompt/completion names second)."""
    values: dict[str, int | float | None] = {}
    for field, keys in _USAGE_KEYS.items():
        key = next((k for k in keys if span.attributes.get(k) is not None), None)
        value = span.attributes.get(key) if key else None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            logger.warning('Ignoring non-numeric usage attribute {} = {!r} on span {}', key, value, span.span_id)
            value = None
        values[field] = value
    return OtelUsage(**values)


def _span_from_orq(raw: dict[str, Any], index: int) -> OtelSpan:
    raw_attrs = raw.get('attributes')
    attributes = flatten_attributes(cast('dict[str, Any]', raw_attrs) if isinstance(raw_attrs, dict) else {})
    messages: dict[str, list[OtelMessage] | None] = {'input_messages': None, 'output_messages': None}
    for field, key in _MESSAGE_ATTRIBUTES:
        if key in attributes and messages[field] is None:
            parsed = _messages_from_attribute(attributes, key)
            if parsed is not None:
                messages[field] = parsed
                del attributes[key]
    system_instructions = None
    if _SYSTEM_INSTRUCTIONS in attributes:
        system_instructions = _parts_from_attribute(attributes, _SYSTEM_INSTRUCTIONS)
        if system_instructions is not None:
            del attributes[_SYSTEM_INSTRUCTIONS]
    _fill_usage_from_summary(attributes, raw)
    error_type = attributes.get('error.type')
    return OtelSpan(
        trace_id=str(raw.get('trace_id') or ''),
        span_id=span_id_of(raw, index),
        parent_span_id=parent_span_id_of(raw),
        name=str(raw.get('name') or ''),
        span_type=raw.get('type') if isinstance(raw.get('type'), str) else None,
        operation=_operation(attributes, raw),
        start_time=parse_time(_first(raw, 'started_at', 'start_time')),
        end_time=parse_time(_first(raw, 'ended_at', 'end_time')),
        status=_status(raw.get('status')),
        error_type=str(error_type) if error_type is not None else None,
        input_messages=messages['input_messages'],
        output_messages=messages['output_messages'],
        system_instructions=system_instructions,
        attributes=attributes,
    )


def _first(raw: dict[str, Any], *keys: str) -> object:
    return next((raw[key] for key in keys if raw.get(key) is not None), None)


def _operation(attributes: dict[str, Any], raw: dict[str, Any]) -> str | None:
    name = attributes.get('gen_ai.operation.name')
    if isinstance(name, str) and name:
        return _OPERATION_ALIASES.get(name, name)
    span_type = raw.get('type')
    return _SPAN_TYPE_OPERATIONS.get(span_type) if isinstance(span_type, str) else None


def _status(value: object) -> Literal['ok', 'error', 'unset']:
    if isinstance(value, dict):
        value = cast('dict[str, Any]', value).get('code')
    text = str(value).lower().removeprefix('status_code_')
    if text in ('ok', 'error', 'unset'):
        return text
    return 'error' if 'error' in text else 'unset'


def _load_json_attribute(attributes: dict[str, Any], key: str) -> tuple[bool, Any]:
    value = attributes[key]
    if not isinstance(value, str):
        return True, value
    try:
        return True, json.loads(value)
    except json.JSONDecodeError as exc:
        logger.warning('Could not parse span attribute {} as JSON ({}); keeping the raw value', key, exc)
        return False, None


def _messages_from_attribute(attributes: dict[str, Any], key: str) -> list[OtelMessage] | None:
    ok, value = _load_json_attribute(attributes, key)
    if not ok:
        return None
    value = _as_list(value)
    if isinstance(value, dict):
        if 'messages' in value:
            value = _as_list(value['messages'])
        elif 'message' in value:
            value = _as_list(value['message'])
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        logger.warning('Span attribute {} is not a message list ({}); keeping the raw value', key, type(value).__name__)
        return None
    messages: list[OtelMessage] = []
    invalid = 0
    for index, message in enumerate(cast('list[Any]', value)):
        parsed, error = _parse_message(message)
        if parsed is None:
            invalid += 1
            logger.warning(
                'Could not parse OTel message {} from span attribute {} ({}); skipping it', index, key, error
            )
        else:
            messages.append(parsed)
    if invalid and not messages:
        logger.warning('No valid OTel messages in span attribute {}; keeping the raw value', key)
        return None
    return messages


def _parse_message(message: Any) -> tuple[OtelMessage | None, ValidationError | None]:
    try:
        return OtelMessage.model_validate(_otel_message(message)), None
    except ValidationError as exc:
        return None, exc


def _parts_from_attribute(attributes: dict[str, Any], key: str) -> list[OtelPart] | None:
    ok, value = _load_json_attribute(attributes, key)
    if not ok:
        return None
    value = _as_list(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        logger.warning('Span attribute {} is not a part list ({}); keeping the raw value', key, type(value).__name__)
        return None
    parts: list[OtelPart] = []
    invalid = 0
    for index, part in enumerate(cast('list[Any]', value)):
        parsed, error = _parse_part(part)
        if parsed is None:
            invalid += 1
            logger.warning('Could not parse OTel part {} from span attribute {} ({}); skipping it', index, key, error)
        else:
            parts.append(parsed)
    if invalid and not parts:
        logger.warning('No valid OTel parts in span attribute {}; keeping the raw value', key)
        return None
    return parts


def _parse_part(part: Any) -> tuple[OtelPart | None, ValidationError | None]:
    try:
        return parse_part(part), None
    except ValidationError as exc:
        return None, exc


def _otel_message(message: Any) -> Any:
    """Return an OTel-parts message; a chat-completions message (`content`, `tool_calls`) is converted."""
    if not isinstance(message, dict) or 'parts' in message:
        return message
    converted = dict(cast('dict[str, Any]', message))
    content = converted.pop('content', None)
    tool_calls = _as_list(converted.pop('tool_calls', None)) or []
    call_id = converted.pop('tool_call_id', None)
    refusal = converted.pop('refusal', None)
    parts: list[Any] = []
    if converted.get('role') == 'tool':
        parts.append({'type': 'tool_call_response', 'id': call_id, 'response': content})
    else:
        parts.extend(_content_parts(content))
    if refusal:
        parts.append({'type': 'refusal', 'content': refusal})
    parts.extend(_tool_call_part(call) for call in (tool_calls if isinstance(tool_calls, list) else [tool_calls]))
    converted['parts'] = parts
    return converted


def _content_parts(content: Any) -> list[Any]:
    if content is None or content == '':
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'content': content}]
    if isinstance(content, list):
        parts: list[Any] = []
        for item in cast('list[Any]', content):
            if isinstance(item, str):
                parts.append({'type': 'text', 'content': item})
            elif isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                parts.append({'type': 'text', 'content': item['text']})
            else:
                parts.append(item)
        return parts
    return [{'type': 'text', 'content': json.dumps(content, default=str)}]


def _tool_call_part(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    call = cast('dict[str, Any]', call)
    function = call.get('function')
    function = cast('dict[str, Any]', function) if isinstance(function, dict) else {}
    arguments = function.get('arguments', call.get('arguments'))
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            arguments = decoded
    return {
        'type': 'tool_call',
        'id': call.get('id'),
        'name': function.get('name', call.get('name')),
        'arguments': arguments,
    }


def _fill_usage_from_summary(attributes: dict[str, Any], raw: dict[str, Any]) -> None:
    """Copy Orq's span-summary `usage` / `cost.total` into `gen_ai.usage.*` where those attributes are absent."""
    summary = raw.get('usage')
    cost = raw.get('cost')
    reported: dict[str, Any] = {}
    if isinstance(summary, dict):
        summary = cast('dict[str, Any]', summary)
        reported = {field: summary.get(key) for field, key in _SUMMARY_USAGE.items()}
    if isinstance(cost, dict):
        reported['cost_usd'] = cast('dict[str, Any]', cost).get('total')
    for field, value in reported.items():
        keys = _USAGE_KEYS[field]
        if value is not None and all(attributes.get(key) is None for key in keys):
            attributes[keys[0]] = value
