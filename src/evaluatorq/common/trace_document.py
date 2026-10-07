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
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluatorq.formats._shared import RAW_ARGUMENTS_EXTRA_KEY, atif_tool_arguments
from evaluatorq.formats.atif import (
    AtifAgent,
    AtifAudioSource,
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
from evaluatorq.types import DataPoint

if TYPE_CHECKING:
    from evaluatorq.trace_finder.models import TraceRecord

_EXTRA_PREFIX = 'evaluatorq.insights.'
_ROLE_SOURCE = {'system': 'system', 'developer': 'system', 'user': 'user'}
_TRAJECTORY_ID_NAMESPACE = 'https://evaluatorq.orq.ai/trace-document/v1/'

LabelSource = Literal['programmatic', 'human', 'judge', 'none']


class DatasetRef(BaseModel):
    """The frozen dataset row a trace document was loaded from."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str | None = None
    row_id: str = Field(min_length=1)


class Outcome(BaseModel):
    """Ground truth for one trajectory: the result, what it means in its dataset, and where it came from.

    `passed` is `None` exactly when `source` is `'none'`, so an unlabelled row has to say so.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    passed: bool | None
    score: float | None = None
    source: LabelSource
    definition: str = Field(min_length=1)

    @model_validator(mode='after')
    def _unlabelled_is_explicit(self) -> Outcome:
        if (self.passed is None) != (self.source == 'none'):
            raise ValueError("passed must be None exactly when source is 'none'")
        return self


class TraceMetadata(BaseModel):
    """Trace identity and source metadata, kept outside the ATIF conversation.

    An Orq trace needs `span_id` and `timestamp`. A dataset row sets `dataset` and `outcome` instead and may
    leave both unset, because benchmark rows rarely record them.
    """

    model_config = ConfigDict(frozen=True, extra='allow')

    schema_version: Literal[1] = 1
    trace_id: str = Field(min_length=1)
    span_id: str | None = Field(default=None, min_length=1)
    timestamp: datetime | None = None
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
    dataset: DatasetRef | None = None
    outcome: Outcome | None = None

    @field_validator('timestamp')
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError('timestamp must include a timezone offset')
        return value

    @model_validator(mode='after')
    def _identifiers_present(self) -> TraceMetadata:
        if self.dataset is None and (not self.span_id or self.timestamp is None):
            raise ValueError('a trace without a dataset needs span_id and timestamp')
        if self.dataset is not None and self.outcome is None:
            raise ValueError("a dataset row needs an outcome; use source='none' when it is unlabelled")
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

    @property
    def source_coverage(self) -> dict[str, Any]:
        """Safe provenance and span-coverage summary for persisted review data."""
        return _source_coverage_projection(self.metadata.capture_metadata)

    def to_datapoint(self) -> DataPoint:
        """Return an evaluatorq `DataPoint` with the trajectory as input and the outcome as expected output."""
        outcome = self.metadata.outcome
        inputs = {'trajectory': self.trajectory.to_json_dict()}
        if self.metadata.dataset is not None:
            inputs['dataset'] = self.metadata.dataset.model_dump(mode='json')
        return DataPoint(
            inputs=inputs,
            expected_output=outcome.model_dump(mode='json') if outcome else None,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward legacy metadata field reads while keeping their storage nested."""
        metadata = self.__dict__.get('metadata')
        if metadata is not None and name in type(metadata).model_fields:
            return getattr(metadata, name)
        raise AttributeError(name)


class TrajectoryCounts(BaseModel):
    """What a dataset parser measured on its source row, compared against the converted trajectory."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_calls: int = Field(ge=0)
    tool_results: int = Field(ge=0)
    reasoning_steps: int = Field(ge=0)


def trajectory_counts(trajectory: AtifTrajectory) -> TrajectoryCounts:
    """Count tool calls, tool results and steps with reasoning, embedded subagents included.

    A dataset parser's test asserts this equals the counts it measured on the raw row, so a call, result or
    reasoning block dropped during conversion fails the test.
    """
    calls = results = reasoning = 0
    for step in trajectory.steps:
        calls += len(step.tool_calls or ())
        results += len(step.observation.results) if step.observation else 0
        reasoning += bool(step.reasoning_content)
    for sub in trajectory.subagent_trajectories or ():
        nested = trajectory_counts(trajectory=sub)
        calls += nested.tool_calls
        results += nested.tool_results
        reasoning += nested.reasoning_steps
    return TrajectoryCounts(tool_calls=calls, tool_results=results, reasoning_steps=reasoning)


def check_trace_document(
    document: TraceDocument,
    *,
    expected_counts: TrajectoryCounts,
    expected_outcome: Outcome,
) -> None:
    """Check source-measured trajectory counts, outcome, and lossless wrapper JSON round-trip."""
    actual_counts = trajectory_counts(trajectory=document.trajectory)
    if actual_counts != expected_counts:
        raise ValueError(f'trajectory counts differ: expected {expected_counts}, got {actual_counts}')
    if document.metadata.outcome != expected_outcome:
        raise ValueError(
            f'trace document outcome differs: expected {expected_outcome}, got {document.metadata.outcome}'
        )

    restored = TraceDocument.model_validate_json(document.model_dump_json())
    if restored != document:
        raise ValueError('trace document changed during JSON round-trip')


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


def _source_coverage_projection(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Keep known provenance and bounded coverage summaries, excluding captured payload sidecars."""
    projected: dict[str, Any] = {}
    for key in ('source', 'start', 'end', 'project_id', 'incomplete_reason'):
        value = capture.get(key)
        if isinstance(value, str):
            projected[key] = value

    raw_coverage = capture.get('signal_span_coverage')
    if not isinstance(raw_coverage, Mapping):
        return projected
    coverage: dict[str, Any] = {}
    for key in ('selected_span_id', 'enrichment_error'):
        value = raw_coverage.get(key)
        if isinstance(value, str):
            coverage[key] = value
    for key in ('selected_span_found', 'selected_data_correlated', 'selected_detail_captured'):
        value = raw_coverage.get(key)
        if isinstance(value, bool):
            coverage[key] = value
    for key in ('scoped_span_count', 'matched_tool_span_count', 'responses_items_captured'):
        value = raw_coverage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            coverage[key] = value
    for key in ('matched_call_ids', 'missing_call_ids', 'enrichment_errors'):
        value = raw_coverage.get(key)
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            coverage[key] = list(value)
    if coverage:
        projected['signal_span_coverage'] = coverage
    return projected


def prompt_messages(document: TraceDocument | TraceRecord) -> list[dict[str, Any]]:
    """Rebuild the historical message dictionaries from ATIF native fields and rendering annotations."""
    doc = ensure_trace_document(document)
    ordered: list[tuple[int, dict[str, Any]]] = []
    for step in doc.trajectory.steps:
        extra = step.extra or {}
        order = _source_order(extra.get(f'{_EXTRA_PREFIX}source_order'), step.step_id - 1)
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
            result_order = _source_order(result_extra.get(f'{_EXTRA_PREFIX}source_order'), order)
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
    raw_orders = selected_span_data.get('step_orders')
    selected_orders = (
        {order for order in raw_orders if isinstance(order, int) and not isinstance(order, bool) and order >= 0}
        if isinstance(raw_orders, list)
        else set()
    )
    legacy_order = selected_span_data.get('step_order')
    if (
        not selected_orders
        and isinstance(legacy_order, int)
        and not isinstance(legacy_order, bool)
        and legacy_order >= 0
    ):
        selected_orders.add(legacy_order)
    representative_order = min(selected_orders) if selected_orders else None
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
        if order in selected_orders:
            raw_items = response_items_by_order.get(str(order))
            if isinstance(raw_items, list) and raw_items:
                extra['evaluatorq.responses_output_items'] = raw_items
            definitions = selected_span_data.get('tool_definitions')
            if selected_span_data.get('tool_definitions_present') and isinstance(definitions, list):
                extra['evaluatorq.responses_tools'] = definitions
        is_representative = order == representative_order
        metrics = selected_span_data.get('metrics') if is_representative else None
        step = AtifStep(
            step_id=len(steps) + 1,
            source=source,
            message=atif_message,
            tool_calls=calls or None,
            timestamp=selected_span_data.get('started_at') if is_representative else None,
            metrics=AtifMetrics.model_validate(metrics) if isinstance(metrics, Mapping) and metrics else None,
            llm_call_count=(1 if is_representative else 0) if order in selected_orders else None,
            extra=(
                {
                    **(extra or {}),
                    'invocation': {
                        key: selected_span_data[key]
                        for key in ('start_timestamp', 'end_timestamp')
                        if selected_span_data.get(key) is not None
                    },
                }
                if is_representative
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
            if _source_order((candidate.extra or {}).get(f'{_EXTRA_PREFIX}source_order'), candidate.step_id - 1) < order
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
            for key, values in call_steps.items():
                call_steps[key] = [replacement if candidate is owner else candidate for candidate in values]

    # Synthetic owners participate in canonical step order too. Sort only
    # after signal enrichment, which indexes the original step list.
    steps.sort(key=lambda step: int((step.extra or {}).get(f'{_EXTRA_PREFIX}source_order', 2**63)))
    steps = [step.model_copy(update={'step_id': index}) for index, step in enumerate(steps, start=1)]

    identity = json.dumps([metadata.trace_id, metadata.span_id], ensure_ascii=False, separators=(',', ':'))
    trajectory_id = str(uuid5(NAMESPACE_URL, f'{_TRAJECTORY_ID_NAMESPACE}{identity}'))
    trajectory = AtifTrajectory(
        trajectory_id=trajectory_id,
        agent=AtifAgent(name=metadata.agent_name or 'unknown', version='unknown', model_name=metadata.model),
        steps=steps or [AtifStep(step_id=1, source='system', message='', extra={f'{_EXTRA_PREFIX}hidden': True})],
    )
    if trajectory.has_audio() and trajectory.schema_version != 'ATIF-v1.8':
        trajectory = trajectory.model_copy(update={'schema_version': 'ATIF-v1.8'})
    return trajectory


def _atif_message(content: Any, extra: dict[str, Any], prefix: str) -> str | list[AtifContentPart]:
    if isinstance(content, str):
        return content
    converted = _atif_content(content, extra, prefix)
    if isinstance(converted, list):
        return converted
    if content is None and not extra.get(f'{prefix}content_present', True):
        return ''
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
            elif isinstance(part, Mapping) and part.get('type') in {'audio', 'input_audio'}:
                audio, audio_layout = _atif_audio(part)
                if audio is None:
                    parts.append(AtifContentPart(type='text', text=''))
                    layout.append({'raw': part})
                else:
                    parts.append(audio)
                    layout.append(audio_layout)
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
            if shape.get('direct'):
                rendered.append({
                    **shape.get('fields', {}),
                    **({'type': shape['type']} if shape.get('type') is not None else {}),
                    str(shape.get('url_key') or 'url'): part.source.path,
                })
            elif shape.get('string_container'):
                rendered.append({
                    **shape.get('fields', {}),
                    'type': shape.get('type'),
                    str(shape.get('container_key')): part.source.path,
                })
            else:
                container = str(shape.get('container_key') or 'image_url')
                url_key = str(shape.get('url_key') or 'url')
                image_value = {**shape.get('container_fields', {}), url_key: part.source.path}
                rendered.append({
                    **shape.get('fields', {}),
                    'type': shape.get('type', 'image_url'),
                    container: image_value,
                })
        elif part.type == 'audio' and isinstance(part.source, AtifAudioSource):
            audio_path = shape.get('raw_data', part.source.path)
            if shape.get('direct'):
                rendered.append({
                    **shape.get('fields', {}),
                    **({'type': shape['type']} if shape.get('type') is not None else {}),
                    str(shape.get('url_key') or 'url'): audio_path,
                })
            elif shape.get('string_container'):
                rendered.append({
                    **shape.get('fields', {}),
                    **({'type': shape['type']} if shape.get('type') is not None else {}),
                    str(shape.get('container_key')): audio_path,
                })
            else:
                rendered.append({
                    **shape.get('fields', {}),
                    **({'type': shape['type']} if shape.get('type') is not None else {}),
                    str(shape.get('container_key') or 'input_audio'): {
                        **shape.get('container_fields', {}),
                        str(shape.get('url_key') or 'url'): audio_path,
                    },
                })
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
    string_container = isinstance(source, str)
    if isinstance(source, Mapping):
        url_key = (
            'url' if isinstance(source.get('url'), str) else 'path' if isinstance(source.get('path'), str) else None
        )
        path = source.get(url_key) if url_key else None
        media_type = source.get('media_type') or source.get('mime_type')
        container_fields = {key: value for key, value in source.items() if key != url_key}
    elif isinstance(source, str):
        url_key = None
        path = source
        media_type = part.get('media_type') or part.get('mime_type')
        container_fields = {}
    else:
        url_key = 'url' if isinstance(part.get('url'), str) else 'path' if isinstance(part.get('path'), str) else None
        path = part.get(url_key) if url_key else None
        media_type = part.get('media_type') or part.get('mime_type')
        container_fields = {}
    if not isinstance(media_type, str):
        media_type = _infer_image_media_type(path) if isinstance(path, str) else None
    if not isinstance(path, str) or media_type not in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}:
        return None, {}
    source_model = AtifImageSource(media_type=media_type, path=path)
    direct = container_key is None
    excluded = {'type', container_key}
    if direct:
        excluded.add(url_key)
    layout = {
        'type': raw_type,
        'container_key': container_key,
        'url_key': url_key,
        'container_fields': container_fields,
        'fields': {key: value for key, value in part.items() if key not in excluded},
        'direct': direct,
        'string_container': string_container,
    }
    return AtifContentPart(type='image', source=source_model), layout


def _infer_image_media_type(path: str) -> str | None:
    """Infer only ATIF-supported image types from data URIs or URL/file suffixes."""
    if path.startswith('data:'):
        header = path[5:].split(',', 1)[0]
        media_type = header.split(';', 1)[0].lower()
    else:
        try:
            url_path = urlsplit(path).path
        except ValueError:
            return None
        suffix = url_path.rsplit('.', 1)[-1].lower() if '.' in url_path else ''
        media_type = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
        }.get(suffix)
    return media_type if media_type in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'} else None


def _atif_audio(part: Mapping[str, Any]) -> tuple[AtifContentPart | None, dict[str, Any]]:
    """Convert URL/path audio parts to ATIF while retaining their prompt layout."""
    container_key = next((key for key in ('input_audio', 'audio', 'source') if key in part), None)
    source = part.get(container_key) if container_key else None
    string_container = isinstance(source, str)
    raw_data = None
    if isinstance(source, Mapping):
        path_key = next((key for key in ('url', 'path', 'data') if isinstance(source.get(key), str)), None)
        path = source.get(path_key) if path_key else None
        raw_type = source.get('media_type') or source.get('mime_type')
        raw_type = raw_type or (f'audio/{source["format"]}' if isinstance(source.get('format'), str) else None)
        extra_fields = {key: value for key, value in source.items() if key != path_key}
        if path_key == 'data' and isinstance(path, str):
            raw_data = path
            try:
                canonical_type = AtifAudioSource(media_type=raw_type, path='placeholder').media_type
            except (ValueError, TypeError):
                return None, {}
            path = path if path.startswith('data:') else f'data:{canonical_type};base64,{path}'
    elif isinstance(source, str):
        path_key, path = 'url', source
        raw_type = part.get('media_type') or part.get('mime_type')
        extra_fields = {}
    else:
        path_key = 'url' if isinstance(part.get('url'), str) else 'path' if isinstance(part.get('path'), str) else None
        path = part.get(path_key) if path_key else None
        raw_type = part.get('media_type') or part.get('mime_type')
        extra_fields = {}
    if not isinstance(path, str) or not isinstance(raw_type, str):
        return None, {}
    try:
        audio_source = AtifAudioSource(media_type=raw_type, path=path)
    except (ValueError, TypeError):
        return None, {}
    direct = container_key is None
    excluded = {'type', container_key}
    if direct:
        excluded.add(path_key)
    layout = {
        'type': part.get('type'),
        'container_key': container_key or 'input_audio',
        'url_key': path_key or 'url',
        'container_fields': extra_fields,
        'fields': {key: value for key, value in part.items() if key not in excluded},
        'direct': direct,
        'string_container': string_container,
    }
    if raw_data is not None:
        layout['raw_data'] = raw_data
    return AtifContentPart(type='audio', source=audio_source), layout


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_order(value: Any, fallback: int) -> int:
    """Read trusted integer ordering metadata, falling back for malformed provenance."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else fallback


def _atif_call(raw: Any, *, fallback_id: str) -> AtifToolCall | None:
    if not isinstance(raw, Mapping):
        return None
    call = dict(raw)
    raw_function = call.get('function')
    function: Mapping[str, Any] = raw_function if isinstance(raw_function, Mapping) else call
    name_keys = [key for key in ('name', 'tool_name') if key in function]
    argument_keys = [key for key in ('arguments', 'input') if key in function]
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
        f'{_EXTRA_PREFIX}name_keys': name_keys,
        f'{_EXTRA_PREFIX}argument_keys': argument_keys,
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
        function_fields = {
            k: v for k, v in raw_function.items() if k not in {'name', 'tool_name', 'arguments', 'input'}
        }
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
        name_keys = extra.get(f'{_EXTRA_PREFIX}name_keys', ['name'])
        argument_keys = extra.get(f'{_EXTRA_PREFIX}argument_keys', ['arguments'])
        for key in name_keys:
            outer[key] = call.function_name
        for key in argument_keys:
            outer[key] = arguments
        return outer
    function = dict(extra.get(f'{_EXTRA_PREFIX}function_fields') or {})
    for key in extra.get(f'{_EXTRA_PREFIX}name_keys', ['name']):
        function[key] = call.function_name
    for key in extra.get(f'{_EXTRA_PREFIX}argument_keys', ['arguments']):
        function[key] = arguments
    outer = dict(extra.get(f'{_EXTRA_PREFIX}call_outer_fields') or {})
    if extra.get(f'{_EXTRA_PREFIX}id_present') or call.tool_call_id != extra.get(f'{_EXTRA_PREFIX}generated_id'):
        outer.setdefault(str(extra.get(f'{_EXTRA_PREFIX}id_key', 'id')), call.tool_call_id)
    outer['function'] = function
    return outer
