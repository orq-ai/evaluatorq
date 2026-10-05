"""Canonical trace document: metadata plus one ATIF conversation chain.

Insights still renders its established chat-message projection for prompts, but
that projection is reconstructed on demand from the ATIF steps and their small
source-format annotations. It is never stored as a second conversation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime  # noqa: TC003 -- Pydantic resolves this annotated model field at runtime.
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluatorq.formats._shared import RAW_ARGUMENTS_EXTRA_KEY, atif_tool_arguments
from evaluatorq.formats.atif import (
    AtifAgent,
    AtifContentPart,
    AtifImageSource,
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifToolCall,
    AtifTrajectory,
)
from evaluatorq.signals.models import (
    SignalReport,  # noqa: TC001 -- Pydantic resolves this annotated model field at runtime.
)
from evaluatorq.signals.preconditions import TRUNCATED_EXTRA_KEY

if TYPE_CHECKING:
    from evaluatorq.trace_finder.models import TraceRecord

_EXTRA_PREFIX = 'evaluatorq.insights.'
_ROLE_SOURCE = {'system': 'system', 'developer': 'system', 'user': 'user'}


class TraceMetadata(BaseModel):
    """Trace identity and source metadata, kept outside the ATIF conversation."""

    model_config = ConfigDict(frozen=True, extra='allow')

    schema_version: Literal[1] = 1
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    timestamp: datetime
    project: str = ''
    model: str = ''
    provider: str = ''
    status: str = ''
    product: str = ''
    trace_type: str = ''
    agent_name: str = ''
    tool_names: tuple[str, ...] = ()
    tool_definition_count: int = Field(default=0, ge=0)
    tool_definition_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    capture_metadata: dict[str, Any] = Field(default_factory=dict)
    signals: SignalReport | None = None

    @field_validator('timestamp')
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('timestamp must include a timezone offset')
        return value

    @model_validator(mode='after')
    def _identifiers_present(self) -> TraceMetadata:
        if not self.trace_id or not self.span_id:
            raise ValueError('trace_id and span_id must be non-empty')
        return self


class TraceDocument(BaseModel):
    """Metadata and a single canonical ATIF trajectory for one trace."""

    model_config = ConfigDict(frozen=True)

    metadata: TraceMetadata
    trajectory: AtifTrajectory

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        """Compatibility view, freshly rendered from the trajectory on access."""
        return tuple(prompt_messages(self))

    def __getattr__(self, name: str) -> Any:
        """Forward legacy metadata field reads while keeping their storage nested."""
        metadata = self.__dict__.get('metadata')
        if metadata is not None and name in type(metadata).model_fields:
            return getattr(metadata, name)
        raise AttributeError(name)


def ensure_trace_document(value: TraceDocument | TraceRecord) -> TraceDocument:
    """Adapt a legacy trace record once, retaining its established prompt projection."""
    if isinstance(value, TraceDocument):
        return value
    record = value.model_dump(mode='python')
    messages = record.pop('messages')
    metadata = TraceMetadata.model_validate(record)
    trajectory = _trajectory_from_messages(metadata, messages)
    transient = {'signal_tool_spans', 'signal_selected_span'}
    cleaned_capture = {key: value for key, value in metadata.capture_metadata.items() if key not in transient}
    cleaned_metadata = metadata.model_copy(update={'capture_metadata': cleaned_capture})
    return TraceDocument(metadata=cleaned_metadata, trajectory=trajectory)


def trace_document_with_signals(document: TraceDocument, report: SignalReport) -> TraceDocument:
    """Return a copy with the computed signal report attached to metadata."""
    if (
        report.trajectory_id
        and document.trajectory.trajectory_id
        and report.trajectory_id != document.trajectory.trajectory_id
    ):
        raise ValueError('signal report trajectory_id does not match the trace document')
    return document.model_copy(update={'metadata': document.metadata.model_copy(update={'signals': report})})


def prompt_messages(document: TraceDocument | TraceRecord) -> list[dict[str, Any]]:
    """Rebuild the historical message dictionaries from ATIF native fields and rendering annotations."""
    doc = ensure_trace_document(document)
    ordered: list[tuple[int, dict[str, Any]]] = []
    for step in doc.trajectory.steps:
        extra = step.extra or {}
        order = int(extra.get(f'{_EXTRA_PREFIX}source_order', step.step_id - 1))
        if extra.get(f'{_EXTRA_PREFIX}hidden'):
            pass
        else:
            role = extra.get(f'{_EXTRA_PREFIX}source_role', 'assistant' if step.source == 'agent' else step.source)
            message: dict[str, Any] = dict(extra.get(f'{_EXTRA_PREFIX}message_fields') or {})
            message['role'] = role
            if f'{_EXTRA_PREFIX}content_override' in extra:
                message['content'] = extra[f'{_EXTRA_PREFIX}content_override']
            elif isinstance(step.message, str):
                if extra.get(f'{_EXTRA_PREFIX}content_present', True):
                    message['content'] = step.message
            else:
                message['content'] = _render_part_list(step.message, extra.get(f'{_EXTRA_PREFIX}content_layout'))
            calls = [_render_call(call) for call in step.tool_calls or []]
            if calls:
                message['tool_calls'] = calls
            ordered.append((order, message))
        for result in step.observation.results if step.observation else []:
            result_extra = result.extra or {}
            if result_extra.get(f'{_EXTRA_PREFIX}hidden'):
                continue
            result_order = int(result_extra.get(f'{_EXTRA_PREFIX}source_order', order))
            result_message = dict(result_extra.get(f'{_EXTRA_PREFIX}message_fields') or {})
            result_message['role'] = result_extra.get(f'{_EXTRA_PREFIX}source_role', 'tool')
            if f'{_EXTRA_PREFIX}content_override' in result_extra:
                result_message['content'] = result_extra[f'{_EXTRA_PREFIX}content_override']
            else:
                result_message['content'] = _render_content(
                    result.content, result_extra.get(f'{_EXTRA_PREFIX}content_layout')
                )
            call_id = result.source_call_id or result_extra.get(f'{_EXTRA_PREFIX}call_id')
            if call_id is not None:
                result_message['tool_call_id'] = call_id
            ordered.append((result_order, result_message))
    ordered.sort(key=itemgetter(0))
    return [message for _, message in ordered]


def _trajectory_from_messages(metadata: TraceMetadata, messages: Any) -> AtifTrajectory:  # noqa: C901 -- preserves the varied source message layouts in one conversion pass.
    if not isinstance(messages, (list, tuple)):
        messages = []
    steps: list[AtifStep] = []
    selected_span_data = _mapping(metadata.capture_metadata.get('signal_selected_span'))
    selected_order = selected_span_data.get('step_order')
    response_items_by_order = _mapping(selected_span_data.get('output_items_by_order'))
    call_steps: dict[str, list[AtifStep]] = {}
    unattached_results: list[tuple[int, dict[str, Any]]] = []
    for order, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            raw = {'role': 'user', 'content': raw}
        item = dict(raw)
        role = item.get('role', 'user')
        if role == 'tool':
            unattached_results.append((order, item))
            continue
        source = (
            cast('Literal["system", "user", "agent"]', _ROLE_SOURCE.get(role, 'agent'))
            if isinstance(role, str)
            else 'agent'
        )
        content = item.get('content')
        extra: dict[str, Any] = {
            f'{_EXTRA_PREFIX}source_order': order,
            f'{_EXTRA_PREFIX}source_role': role,
            f'{_EXTRA_PREFIX}content_present': 'content' in item,
        }
        message_fields = {k: v for k, v in item.items() if k not in {'role', 'content', 'tool_calls'}}
        if message_fields:
            extra[f'{_EXTRA_PREFIX}message_fields'] = message_fields
        atif_message = _atif_message(content, extra, _EXTRA_PREFIX)
        calls = [
            _atif_call(call, fallback_id=f'unknown-{order}-{index}')
            for index, call in enumerate(item.get('tool_calls') or [])
        ]
        calls = [call for call in calls if call is not None]
        if order == selected_order:
            raw_items = response_items_by_order.get(str(order))
            if isinstance(raw_items, list) and raw_items:
                extra['evaluatorq.responses_output_items'] = raw_items
            definitions = selected_span_data.get('tool_definitions')
            if selected_span_data.get('tool_definitions_present') and isinstance(definitions, list):
                extra['evaluatorq.responses_tools'] = definitions
        metrics = selected_span_data.get('metrics') if order == selected_order else None
        step = AtifStep(
            step_id=len(steps) + 1,
            source=source,
            message=atif_message,
            tool_calls=calls or None,
            timestamp=selected_span_data.get('started_at') if order == selected_order else None,
            metrics=AtifMetrics.model_validate(metrics) if isinstance(metrics, Mapping) and metrics else None,
            llm_call_count=1 if order == selected_order else None,
            extra=(
                {
                    **(extra or {}),
                    'invocation': {
                        key: selected_span_data[key]
                        for key in ('start_timestamp', 'end_timestamp')
                        if selected_span_data.get(key) is not None
                    },
                }
                if order == selected_order
                and (
                    selected_span_data.get('start_timestamp') is not None
                    or selected_span_data.get('end_timestamp') is not None
                )
                else extra or None
            ),
        )
        steps.append(step)
        for call in calls:
            call_steps.setdefault(call.tool_call_id, []).append(step)

    # Attach each result to its matching call step so ATIF can analyze the
    # action/result pair. The independent source-order selector restores the
    # exact former chat projection at rendering time.
    for order, item in unattached_results:
        call_id = item.get('tool_call_id') or item.get('call_id')
        candidates = call_steps.get(call_id, []) if isinstance(call_id, str) else []
        prior = [
            candidate
            for candidate in candidates
            if int((candidate.extra or {}).get(f'{_EXTRA_PREFIX}source_order', -1)) < order
            and not any(
                result.source_call_id == call_id
                for result in (candidate.observation.results if candidate.observation else [])
            )
        ]
        owner = prior[-1] if prior else None
        if owner is None:
            owner = AtifStep(
                step_id=len(steps) + 1,
                source='system',
                message='',
                extra={f'{_EXTRA_PREFIX}hidden': True, f'{_EXTRA_PREFIX}source_order': order},
            )
            steps.append(owner)
        result_extra: dict[str, Any] = {
            f'{_EXTRA_PREFIX}source_order': order,
            f'{_EXTRA_PREFIX}source_role': 'tool',
        }
        if call_id is not None:
            result_extra[f'{_EXTRA_PREFIX}call_id'] = call_id
        result_fields = {k: v for k, v in item.items() if k not in {'role', 'content', 'tool_call_id', 'call_id'}}
        if result_fields:
            result_extra[f'{_EXTRA_PREFIX}message_fields'] = result_fields
        result_content = _atif_content(item.get('content'), result_extra, _EXTRA_PREFIX)
        source_result = _mapping(item.get('trace_finder_metadata')).get('tool_result')
        source_result = source_result if isinstance(source_result, Mapping) else {}
        for key in ('status', 'error_type'):
            value = source_result.get(key, item.get(key))
            if value is not None:
                if (
                    key == 'status'
                    and isinstance(value, str)
                    and any(word in value.casefold() for word in ('error', 'fail'))
                ):
                    value = 'error'
                result_extra[key] = value
        if item.get(TRUNCATED_EXTRA_KEY) or source_result.get(TRUNCATED_EXTRA_KEY):
            result_extra[TRUNCATED_EXTRA_KEY] = True
        known_call_ids = {call.tool_call_id for call in owner.tool_calls or []}
        source_call_id = call_id if isinstance(call_id, str) and call_id in known_call_ids else None
        if source_call_id is not None:
            result_extra.pop(f'{_EXTRA_PREFIX}call_id', None)
        result = AtifObservationResult(source_call_id=source_call_id, content=result_content, extra=result_extra)
        updated = [*(owner.observation.results if owner.observation else []), result]
        replacement = owner.model_copy(update={'observation': AtifObservation(results=updated)})
        steps[owner.step_id - 1] = replacement
        for key, values in call_steps.items():
            call_steps[key] = [replacement if candidate is owner else candidate for candidate in values]

    unmapped_items = selected_span_data.get('unmapped_output_items')
    if isinstance(unmapped_items, list) and unmapped_items:
        steps.append(
            AtifStep(
                step_id=len(steps) + 1,
                source='system',
                message='',
                extra={
                    f'{_EXTRA_PREFIX}hidden': True,
                    'evaluatorq.responses_output_items': unmapped_items,
                },
            )
        )

    signal_spans = metadata.capture_metadata.get('signal_tool_spans', {})
    if isinstance(signal_spans, Mapping):
        for call_id, fields in signal_spans.items():
            owners = call_steps.get(call_id, [])
            if not owners or not isinstance(fields, Mapping):
                continue
            owner = owners[-1]
            observation = owner.observation
            results = list(observation.results if observation else [])
            paired_index = next((i for i, result in enumerate(results) if result.source_call_id == call_id), None)
            signal_fields = {
                key: value
                for key, value in fields.items()
                if key in {'status', 'error_type', 'start_timestamp', 'end_timestamp'}
            }
            if not signal_fields:
                continue
            if paired_index is None:
                results.append(
                    AtifObservationResult(
                        source_call_id=call_id,
                        content=None,
                        extra={**signal_fields, f'{_EXTRA_PREFIX}hidden': True},
                    )
                )
            else:
                result = results[paired_index]
                result_extra = {**(result.extra or {})}
                result_extra.update({key: value for key, value in signal_fields.items() if key not in result_extra})
                results[paired_index] = result.model_copy(update={'extra': result_extra})
            replacement = owner.model_copy(update={'observation': AtifObservation(results=results)})
            steps[owner.step_id - 1] = replacement
            call_steps[call_id] = [replacement if candidate is owner else candidate for candidate in owners]

    trajectory_id = metadata.trace_id or 'trace'
    return AtifTrajectory(
        trajectory_id=trajectory_id,
        agent=AtifAgent(name=metadata.agent_name or 'unknown', version='unknown', model_name=metadata.model),
        steps=steps or [AtifStep(step_id=1, source='system', message='', extra={f'{_EXTRA_PREFIX}hidden': True})],
    )


def _atif_message(content: Any, extra: dict[str, Any], prefix: str) -> str | list[AtifContentPart]:
    if isinstance(content, str):
        return content
    converted = _atif_content(content, extra, prefix)
    if isinstance(converted, list):
        return converted
    extra[f'{prefix}content_override'] = content
    return ''


def _atif_content(
    content: Any, extra: dict[str, Any], prefix: str = _EXTRA_PREFIX
) -> str | list[AtifContentPart] | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[AtifContentPart] = []
        layout: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                parts.append(AtifContentPart(type='text', text=part))
                layout.append({'string': True})
            elif isinstance(part, Mapping) and part.get('type') in {
                'text',
                'input_text',
                'output_text',
                'summary_text',
                'refusal',
            }:
                text_key = next((key for key in ('text', 'content', 'refusal') if key in part), None)
                text = part.get(text_key) if text_key else None
                if isinstance(text, str):
                    parts.append(AtifContentPart(type='text', text=text))
                    layout.append({
                        'string': False,
                        'type': part.get('type'),
                        'text_key': text_key,
                        'fields': {k: v for k, v in part.items() if k not in {'type', text_key}},
                    })
                else:
                    parts.append(AtifContentPart(type='text', text=''))
                    layout.append({'raw': part})
            elif isinstance(part, Mapping) and part.get('type') in {'image', 'image_url', 'input_image'}:
                image, image_layout = _atif_image(part)
                if image is None:
                    parts.append(AtifContentPart(type='text', text=''))
                    layout.append({'raw': part})
                else:
                    parts.append(image)
                    layout.append(image_layout)
            else:
                parts.append(AtifContentPart(type='text', text=''))
                layout.append({'raw': part})
        extra[f'{prefix}content_layout'] = layout
        return parts
    extra[f'{prefix}content_override'] = content
    return None


def _render_content(content: Any, layout: Any = None) -> Any:
    if content is None or isinstance(content, str):
        return content
    return _render_part_list(content, layout)


def _render_part_list(content: Any, layout: Any = None) -> list[Any]:
    rendered: list[Any] = []
    for index, part in enumerate(content):
        if not isinstance(part, AtifContentPart):
            rendered.append(part)
            continue
        shape = (
            layout[index]
            if isinstance(layout, list) and index < len(layout) and isinstance(layout[index], Mapping)
            else {}
        )
        if 'raw' in shape:
            rendered.append(shape['raw'])
            continue
        if shape.get('string'):
            rendered.append(part.text or '')
        elif part.type == 'image' and isinstance(part.source, AtifImageSource):
            container = str(shape.get('container_key') or 'image_url')
            url_key = str(shape.get('url_key') or 'url')
            image_value = {**shape.get('container_fields', {}), url_key: part.source.path}
            rendered.append({**shape.get('fields', {}), 'type': shape.get('type', 'image_url'), container: image_value})
        else:
            rendered.append({
                **shape.get('fields', {}),
                **({'type': shape['type']} if shape.get('type') is not None else {}),
                str(shape.get('text_key') or 'text'): part.text or '',
            })
    return rendered


def _atif_image(part: Mapping[str, Any]) -> tuple[AtifContentPart | None, dict[str, Any]]:
    raw_type = part.get('type')
    container_key = 'image_url' if 'image_url' in part else 'source' if 'source' in part else None
    source = part.get(container_key) if container_key else None
    if isinstance(source, Mapping):
        url_key = (
            'url' if isinstance(source.get('url'), str) else 'path' if isinstance(source.get('path'), str) else None
        )
        path = source.get(url_key) if url_key else None
        media_type = source.get('media_type') or source.get('mime_type')
        container_fields = {key: value for key, value in source.items() if key != url_key}
    else:
        url_key = 'url' if isinstance(part.get('url'), str) else 'path' if isinstance(part.get('path'), str) else None
        path = part.get(url_key) if url_key else None
        media_type = part.get('media_type') or part.get('mime_type')
        container_fields = {}
    if not isinstance(path, str) or media_type not in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}:
        return None, {}
    source_model = AtifImageSource(media_type=media_type, path=path)
    layout = {
        'type': raw_type,
        'container_key': container_key,
        'url_key': url_key,
        'container_fields': container_fields,
        'fields': {key: value for key, value in part.items() if key not in {'type', container_key}},
    }
    return AtifContentPart(type='image', source=source_model), layout


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _atif_call(raw: Any, *, fallback_id: str) -> AtifToolCall | None:
    if not isinstance(raw, Mapping):
        return None
    call = dict(raw)
    raw_function = call.get('function')
    function: Mapping[str, Any] = raw_function if isinstance(raw_function, Mapping) else call
    name = function.get('name') or function.get('tool_name') or '(unnamed)'
    raw_arguments = function.get('arguments', function.get('input', {}))
    arguments, raw_argument_provenance = atif_tool_arguments(raw_arguments, name)
    call_id = call.get('id') or call.get('call_id') or call.get('item_id')
    id_present = isinstance(call_id, str) and bool(call_id)
    if not id_present:
        call_id = fallback_id
    nested = isinstance(call.get('function'), Mapping)
    extra: dict[str, Any] = {
        f'{_EXTRA_PREFIX}nested_function': nested,
        f'{_EXTRA_PREFIX}id_key': 'id' if 'id' in call else 'call_id' if 'call_id' in call else 'item_id',
        f'{_EXTRA_PREFIX}id_present': id_present,
        f'{_EXTRA_PREFIX}generated_id': call_id,
    }
    consumed = {'id', 'call_id', 'item_id'}
    if nested:
        consumed.add('function')
    else:
        consumed.update({'name', 'arguments', 'input', 'tool_name'})
    extra[f'{_EXTRA_PREFIX}call_outer_fields'] = {k: v for k, v in call.items() if k not in consumed}
    if isinstance(raw_arguments, str):
        extra[f'{_EXTRA_PREFIX}arguments_text'] = raw_arguments
    elif not isinstance(raw_arguments, dict):
        extra[f'{_EXTRA_PREFIX}arguments_value'] = raw_arguments
    if raw_argument_provenance is not None:
        extra[RAW_ARGUMENTS_EXTRA_KEY] = raw_argument_provenance
    if isinstance(raw_function, Mapping):
        function_fields = {k: v for k, v in raw_function.items() if k not in {'name', 'arguments'}}
        if function_fields:
            extra[f'{_EXTRA_PREFIX}function_fields'] = function_fields
    return AtifToolCall(
        tool_call_id=call_id,
        function_name=str(name),
        arguments=arguments,
        extra=extra or None,
    )


def _render_call(call: AtifToolCall) -> dict[str, Any]:
    extra = call.extra or {}
    raw_arguments = extra.get(f'{_EXTRA_PREFIX}arguments_text')
    arguments: Any = call.arguments
    if f'{_EXTRA_PREFIX}arguments_value' in extra and not call.arguments:
        arguments = extra[f'{_EXTRA_PREFIX}arguments_value']
    if isinstance(raw_arguments, str):
        try:
            decoded_arguments = json.loads(raw_arguments)
            if decoded_arguments == call.arguments or (not isinstance(decoded_arguments, dict) and not call.arguments):
                arguments = raw_arguments
        except (TypeError, ValueError):
            if not call.arguments:
                arguments = raw_arguments
    if not extra.get(f'{_EXTRA_PREFIX}nested_function'):
        outer = dict(extra.get(f'{_EXTRA_PREFIX}call_outer_fields') or {})
        id_key = extra.get(f'{_EXTRA_PREFIX}id_key', 'id')
        if extra.get(f'{_EXTRA_PREFIX}id_present') or call.tool_call_id != extra.get(f'{_EXTRA_PREFIX}generated_id'):
            outer[id_key] = call.tool_call_id
        if 'tool_name' in outer:
            outer['tool_name'] = call.function_name
        else:
            outer['name'] = call.function_name
        if 'input' in outer:
            outer['input'] = arguments
        else:
            outer['arguments'] = arguments
        return outer
    function = {
        **(extra.get(f'{_EXTRA_PREFIX}function_fields') or {}),
        'name': call.function_name,
        'arguments': arguments,
    }
    outer = dict(extra.get(f'{_EXTRA_PREFIX}call_outer_fields') or {})
    if extra.get(f'{_EXTRA_PREFIX}id_present') or call.tool_call_id != extra.get(f'{_EXTRA_PREFIX}generated_id'):
        outer.setdefault(str(extra.get(f'{_EXTRA_PREFIX}id_key', 'id')), call.tool_call_id)
    outer['function'] = function
    return outer
