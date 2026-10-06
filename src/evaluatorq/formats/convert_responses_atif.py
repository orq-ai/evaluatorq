"""Convert between Responses transcripts and ATIF trajectories.

`items` are authoritative: steps are cut from them. Each `Response` starts a new agent step at its first
output item and enriches it with model, usage, timing and finish status. `atif_to_responses` emits one
`Response` per agent step whose `output` holds that step's output items, so consecutive agent steps with no
tool result between them come back as separate steps.
"""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from loguru import logger
from openai.types.responses import Response, ResponseError, ResponseOutputItem, ResponseStatus
from openai.types.responses.response import IncompleteDetails
from pydantic import BaseModel, ValidationError

from evaluatorq.contracts import (
    ContentPart,
    FunctionCall,
    InputImageContent,
    InputTextContent,
    Message,
    StrategyToolCall,
    content_to_text,
    tool_result_to_text,
)
from evaluatorq.formats._ids import content_seed, stable_hex
from evaluatorq.formats._shared import (
    COMPACTION_EXTRA_KEY,
    RAW_ARGUMENTS_EXTRA_KEY,
    atif_content_text,
    atif_tool_arguments,
    compaction_extra,
    final_metrics,
    join_text,
    json_arguments_text,
    media_marker,
    parse_iso,
    part_text,
)
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
from evaluatorq.formats.responses import (
    ResponsesConversation,
    is_output_item,
    output_item,
    response_starts,
    walk_items,
)
from evaluatorq.openresponses.input_items import messages_to_responses_input

if TYPE_CHECKING:
    from openai.types.responses import ResponseUsage

_TEXT_PART_TYPES = frozenset({'input_text', 'output_text', 'text'})
_UNMAPPED_OUTPUTS_KEY = 'evaluatorq.responses_output_items'
_UNMAPPED_RESULTS_KEY = 'evaluatorq.responses_result_items'
_RESULT_ORDER_KEY = 'evaluatorq.responses_result_order'
_RESPONSE_TOOLS_KEY = 'evaluatorq.responses_tools'
_IMAGE_MEDIA_TYPES: dict[str, Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']] = {
    'image/jpeg': 'image/jpeg',
    'image/png': 'image/png',
    'image/gif': 'image/gif',
    'image/webp': 'image/webp',
}
_RESPONSE_STATUSES: frozenset[str] = frozenset(get_args(ResponseStatus))
_NO_STEPS = 'ResponsesConversation has no items; ATIF needs at least one step'
# Response.metadata key listing the ATIF token counts that were unset (Responses usage has no null counts).
UNSET_USAGE_KEY = 'atif_unset_usage'
_UNMAPPED_METRICS = ('cost_usd', 'prompt_token_ids', 'completion_token_ids', 'logprobs')


@dataclass
class _Draft:
    """One ATIF step being assembled from consecutive Responses items."""

    source: Literal['system', 'user', 'agent']
    texts: list[str] = field(default_factory=list)
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    unmapped_outputs: list[dict[str, Any]] = field(default_factory=list)
    unmapped_results: list[dict[str, Any]] = field(default_factory=list)
    result_order: list[dict[str, int | str]] = field(default_factory=list)
    seen_result: bool = False
    item: dict[str, Any] | None = None  # the message item of a user/system step
    response: Response | None = None  # the model call that produced an agent step


# Responses -> ATIF


def responses_to_atif(
    conv: ResponsesConversation,
    *,
    agent_name: str = 'unknown',
    agent_version: str = 'unknown',
    session_id: str | None = None,
) -> AtifTrajectory:
    """Build an ATIF trajectory from Responses items; see `ResponsesConversation.to_atif` for the mapping."""
    drafts = _segment(conv.items, response_starts(conv.items, conv.responses or []))
    if not drafts:
        raise ValueError(_NO_STEPS)
    steps = [
        _agent_step(draft, index + 1) if draft.source == 'agent' else _user_step(draft, index + 1)
        for index, draft in enumerate(drafts)
    ]
    return AtifTrajectory(
        session_id=session_id or stable_hex(content_seed(conv.items), length=16),
        agent=AtifAgent(name=agent_name, version=agent_version),
        steps=steps,
        final_metrics=final_metrics(steps),
    )


def _segment(items: list[dict[str, Any]], starts: dict[int, Response]) -> list[_Draft]:
    """Cut Responses items into step drafts: user/system messages, and agent turns closed by a tool result or
    by the first output item of the next `Response` (`starts` maps item indices to responses)."""
    drafts: list[_Draft] = []

    def current_agent(index: int) -> _Draft:
        last = drafts[-1] if drafts else None
        if last is None or last.source != 'agent' or last.seen_result or index in starts:
            last = _Draft(source='agent', response=starts.get(index))
            drafts.append(last)
        return last

    def message(index: int, item: dict[str, Any]) -> None:
        role = item.get('role')
        if role in ('user', 'system', 'developer'):
            drafts.append(_Draft(source='user' if role == 'user' else 'system', item=item))
        elif role == 'assistant':
            current_agent(index).texts.append(_assistant_text(item.get('content')))
        else:
            logger.warning('Skipping Responses message with role {!r}: ATIF has no such step source.', role)

    def call(index: int, item: dict[str, Any]) -> None:
        if isinstance(item.get('call_id'), str) and item['call_id']:
            current_agent(index).calls.append(item)
        else:
            logger.warning('Responses function_call {!r} has no call_id; skipping it.', item.get('name'))

    def custom_call(index: int, item: dict[str, Any]) -> None:
        if isinstance(item.get('call_id'), str) and item['call_id']:
            current_agent(index).unmapped_outputs.append(item)
        else:
            logger.warning('Responses custom_tool_call {!r} has no call_id; skipping it.', item.get('name'))

    def unsupported_tool_call(index: int, item: dict[str, Any]) -> None:
        """Keep unsupported Responses tool activity available for roundtrip and signal coverage checks."""
        current_agent(index).unmapped_outputs.append(item)

    handlers = {
        'message': message,
        'reasoning': lambda index, item: current_agent(index).reasoning.append(item),
        'function_call': call,
        'custom_tool_call': custom_call,
        'mcp_call': lambda index, item: current_agent(index).unmapped_outputs.append(item),
        'function_call_output': lambda _, item: _attach_output(item, drafts),
        'custom_tool_call_output': lambda _, item: _attach_custom_output(item, drafts),
        'compaction': lambda _, item: drafts.append(_Draft(source='system', item=item)),
    }
    for item in items:
        kind = item.get('type')
        if isinstance(kind, str) and kind.endswith('_call') and kind not in handlers:
            handlers[kind] = unsupported_tool_call
    walk_items(items, handlers, 'ATIF')
    return drafts


def _attach_custom_output(item: dict[str, Any], drafts: list[_Draft]) -> None:
    call_id = item.get('call_id')
    for draft in reversed(drafts):
        if draft.source != 'agent':
            break
        if any(call.get('call_id') == call_id for call in draft.unmapped_outputs):
            draft.unmapped_results.append(item)
            draft.result_order.append({'type': 'custom', 'index': len(draft.unmapped_results) - 1})
            draft.seen_result = True
            return
    logger.warning('custom_tool_call_output for call_id {!r} matches no custom_tool_call; preserving it.', call_id)
    last = drafts[-1] if drafts and drafts[-1].source == 'agent' else _Draft(source='agent')
    if not drafts or last is not drafts[-1]:
        drafts.append(last)
    last.unmapped_results.append(item)
    last.result_order.append({'type': 'custom', 'index': len(last.unmapped_results) - 1})
    last.seen_result = True


def _attach_output(item: dict[str, Any], drafts: list[_Draft]) -> None:
    call_id = item.get('call_id')
    output = item.get('output')
    content = _tool_output_content(output)
    for draft in reversed(drafts):
        if draft.source != 'agent':
            break
        if any(c.get('call_id') == call_id for c in draft.calls):
            draft.results.append({'source_call_id': call_id, 'content': content})
            draft.result_order.append({'type': 'function', 'index': len(draft.results) - 1})
            draft.seen_result = True
            return
    last = drafts[-1] if drafts else None
    if last is None or last.source != 'agent':
        logger.warning(
            'function_call_output for call_id {!r} matches no function_call and follows no agent turn; '
            'starting an empty agent step for it.',
            call_id,
        )
        last = _Draft(source='agent')
        drafts.append(last)
    else:
        logger.warning(
            'function_call_output for call_id {!r} matches no function_call; attaching it to the last agent step '
            'with no source_call_id.',
            call_id,
        )
    last.results.append({'source_call_id': None, 'content': content, 'extra': {'orphan_call_id': call_id}})
    last.result_order.append({'type': 'function', 'index': len(last.results) - 1})
    last.seen_result = True


def _tool_output_content(output: Any) -> str | list[AtifContentPart] | None:
    """Keep typed Responses tool output as ATIF text/image parts where its schema allows it."""
    if isinstance(output, str) or output is None:
        return output
    if not isinstance(output, list):
        return tool_result_to_text(output)
    parts = [_user_part(part) for part in output]
    if all(part.type == 'text' for part in parts):
        return join_text(part.text or '' for part in parts)
    return parts


def _assistant_text(content: Any) -> str:
    """Flatten an assistant message item's content to text."""
    if content is None or isinstance(content, str):
        return content_to_text(content)
    if not isinstance(content, list):
        logger.warning('Assistant message content is {}; dropping it.', type(content).__name__)
        return ''
    texts: list[str] = []
    for part in content:
        part_type = part.get('type') if isinstance(part, dict) else None
        if part_type in _TEXT_PART_TYPES or part_type == 'refusal':
            texts.append(part_text(part, 'Assistant message'))
        else:
            logger.warning('Dropping assistant content part of type {!r}: ATIF agent messages are text.', part_type)
    return join_text(texts)


def _user_step(draft: _Draft, step_id: int) -> AtifStep:
    item = draft.item or {}
    if item.get('type') == 'compaction':
        return AtifStep(step_id=step_id, source='system', message='', extra=compaction_extra([item]))
    extra = {'original_role': 'developer'} if item.get('role') == 'developer' else None
    return AtifStep(step_id=step_id, source=draft.source, message=_user_content(item.get('content')), extra=extra)


def _user_content(content: Any) -> str | list[AtifContentPart]:
    """Map user/system message content to an ATIF message; all-text content collapses to a string."""
    if content is None or isinstance(content, str):
        return content or ''
    if not isinstance(content, list):
        logger.warning('Message content is {}; dropping it.', type(content).__name__)
        return ''
    parts = [_user_part(part) for part in content]
    if all(p.type == 'text' for p in parts):
        return join_text(p.text or '' for p in parts)
    return parts


def _user_part(part: Any) -> AtifContentPart:
    part_type = part.get('type') if isinstance(part, dict) else type(part).__name__
    if isinstance(part, dict) and part_type in _TEXT_PART_TYPES:
        return AtifContentPart(type='text', text=part_text(part, 'Message'))
    if isinstance(part, dict) and part_type == 'input_image' and isinstance(part.get('image_url'), str):
        url: str = part['image_url']
        return AtifContentPart(type='image', source=AtifImageSource(media_type=_image_media_type(url), path=url))
    name = (part.get('filename') or part.get('file_id') or part.get('file_url')) if isinstance(part, dict) else None
    logger.warning('Content part of type {!r} has no ATIF equivalent; rendering a [file: ...] marker.', part_type)
    return AtifContentPart(type='text', text=f'[file: {name or part_type}]')


def _image_media_type(url: str) -> Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']:
    if url.startswith('data:'):
        guessed = url[5:].split(';', 1)[0].split(',', 1)[0]
    else:
        guessed = mimetypes.guess_type(url.split('?', 1)[0])[0] or ''
    if (media_type := _IMAGE_MEDIA_TYPES.get(guessed)) is not None:
        return media_type
    logger.warning('Cannot tell the image media type of {!r}; assuming image/png.', url[:80])
    return 'image/png'


def _agent_step(draft: _Draft, step_id: int) -> AtifStep:
    response = draft.response
    extra: dict[str, Any] = {}
    reasoning_texts: list[str] = []
    for item in draft.reasoning:
        texts = [
            str(p.get('text') or '')
            for key in ('summary', 'content')
            for p in item.get(key) or []
            if isinstance(p, dict) and p.get('text')
        ]
        if not texts and item.get('encrypted_content'):
            extra['encrypted_reasoning'] = True
        reasoning_texts.extend(texts)
    tool_calls = [_tool_call(call) for call in draft.calls]
    fc_item_ids = {c['call_id']: c['id'] for c in draft.calls if isinstance(c.get('id'), str)}
    if fc_item_ids:
        extra['fc_item_ids'] = fc_item_ids
    if draft.unmapped_outputs:
        extra[_UNMAPPED_OUTPUTS_KEY] = draft.unmapped_outputs
    if draft.unmapped_results:
        extra[_UNMAPPED_RESULTS_KEY] = draft.unmapped_results
    result_types = {entry['type'] for entry in draft.result_order}
    if len(result_types) > 1:
        extra[_RESULT_ORDER_KEY] = draft.result_order
    results = [AtifObservationResult(**result) for result in draft.results]
    fields: dict[str, Any] = {}
    if response is not None:
        placeholder = _is_placeholder(response)
        extra.update(_response_extra(response, placeholder=placeholder))
        if not placeholder:
            response_tools = [tool.model_dump(mode='json', exclude_none=True) for tool in response.tools]
            if response_tools:
                extra[_RESPONSE_TOOLS_KEY] = response_tools
            fields = {
                'model_name': response.model or None,
                'timestamp': (
                    datetime.fromtimestamp(response.created_at, tz=timezone.utc).isoformat()
                    if response.created_at
                    else None
                ),
                'metrics': _metrics_from_usage(response.usage, _unset_usage(response)),
                'llm_call_count': 1,
            }
    return AtifStep(
        step_id=step_id,
        source='agent',
        message=join_text(draft.texts),
        reasoning_content='\n'.join(reasoning_texts) or None,
        tool_calls=tool_calls or None,
        observation=AtifObservation(results=results) if results else None,
        extra=extra or None,
        **fields,
    )


def _tool_call(call: dict[str, Any]) -> AtifToolCall:
    arguments, raw_arguments = atif_tool_arguments(call.get('arguments'), call.get('name'))
    return AtifToolCall(
        tool_call_id=call['call_id'],
        function_name=str(call.get('name') or ''),
        arguments=arguments,
        extra={RAW_ARGUMENTS_EXTRA_KEY: raw_arguments} if raw_arguments is not None else None,
    )


def _is_placeholder(response: Response) -> bool:
    """True for the stand-in `atif_to_responses` writes for a step with no model, usage or timestamp."""
    return response.model == '' and response.usage is None and not response.created_at


def _unset_usage(response: Response) -> frozenset[str]:
    names = (response.metadata or {}).get(UNSET_USAGE_KEY) or ''
    return frozenset(name for name in names.split(',') if name)


def _response_extra(response: Response, *, placeholder: bool) -> dict[str, Any]:
    """Response fields kept in step `extra`; a placeholder's made-up id is not one of them.

    The outcome becomes `extra.error_type` and `extra.finish_reasons`, the keys OTel reads. `status`, `error`
    and `incomplete_details` are kept as well only when they say more than those two keys reproduce.
    """
    extra: dict[str, Any] = {} if placeholder else {'response_id': response.id}
    error_type, finish_reasons = _read_outcome(
        response.error.code if response.error else None,
        response.incomplete_details.reason if response.incomplete_details else None,
        response.metadata,
    )
    if error_type is not None:
        extra['error_type'] = error_type
    if finish_reasons is not None:
        extra['finish_reasons'] = finish_reasons
    derived = _outcome_fields(error_type, finish_reasons, failed=error_type is not None)
    if response.status is not None and response.status != derived['status']:
        extra['status'] = response.status
    error = response.error.model_dump(mode='json') if response.error is not None else None
    if error != derived['error']:
        extra['error'] = error
    incomplete = response.incomplete_details.model_dump(mode='json') if response.incomplete_details else None
    if incomplete != derived['incomplete_details']:
        extra['incomplete_details'] = incomplete
    return {key: value for key, value in extra.items() if value is not None}


# A step's outcome is its error type and finish reasons (the OTel `error.type` and
# `gen_ai.response.finish_reasons`). A Response holds it as `status`, `error` and `incomplete_details`, whose
# values are closed enums; what they cannot hold goes to `Response.metadata`, which wins on the way back.
ERROR_TYPE_KEY = 'atif_error_type'
FINISH_REASONS_KEY = 'atif_finish_reasons'
_ERROR_CODES: frozenset[str] = frozenset(get_args(ResponseError.model_fields['code'].annotation))
_INCOMPLETE_REASONS = {'length': 'max_output_tokens', 'content_filter': 'content_filter'}
_FINISH_REASONS = {reason: finish for finish, reason in _INCOMPLETE_REASONS.items()}


def _outcome_fields(error_type: str | None, finish_reasons: list[str] | None, *, failed: bool) -> dict[str, Any]:
    """The Response `status`, `error` and `incomplete_details` that state an outcome."""
    if failed:
        code = error_type if error_type in _ERROR_CODES else 'server_error'
        error = {'code': code, 'message': error_type or 'error'}
        return {'status': 'failed', 'error': error, 'incomplete_details': None}
    reason = next((_INCOMPLETE_REASONS[f] for f in finish_reasons or [] if f in _INCOMPLETE_REASONS), None)
    if reason is not None:
        return {'status': 'incomplete', 'error': None, 'incomplete_details': {'reason': reason}}
    return {'status': 'completed', 'error': None, 'incomplete_details': None}


def _read_outcome(
    error_code: str | None, incomplete_reason: str | None, metadata: dict[str, str] | None
) -> tuple[str | None, list[str] | None]:
    """Error type and finish reasons from a Response's fields, overridden by the metadata keys."""
    error_type = error_code
    finish_reasons = [_FINISH_REASONS[incomplete_reason]] if incomplete_reason in _FINISH_REASONS else None
    metadata = metadata or {}
    if ERROR_TYPE_KEY in metadata:
        error_type = metadata[ERROR_TYPE_KEY] or None
    if FINISH_REASONS_KEY in metadata:
        try:
            stored = json.loads(metadata[FINISH_REASONS_KEY])
        except json.JSONDecodeError:
            stored = None
        if stored is None or (isinstance(stored, list) and all(isinstance(f, str) for f in stored)):
            finish_reasons = cast('list[str] | None', stored)
        else:
            logger.warning('Response metadata {} is not a list of strings; ignoring it', FINISH_REASONS_KEY)
    return error_type, finish_reasons


def _step_outcome(step: AtifStep) -> tuple[str | None, list[str] | None, bool]:
    """Error type, finish reasons, and whether the call failed (an error type, an `error` finish reason, or
    an OTel invocation status of `error`)."""
    extra = step.extra or {}
    error_type = extra.get('error_type')
    if error_type is not None and not isinstance(error_type, str):
        _warn_foreign('error_type', step.step_id, error_type)
        error_type = None
    finish_reasons = extra.get('finish_reasons')
    if finish_reasons is not None and not (
        isinstance(finish_reasons, list) and all(isinstance(f, str) for f in finish_reasons)
    ):
        _warn_foreign('finish_reasons', step.step_id, finish_reasons)
        finish_reasons = None
    finish_reasons = cast('list[str] | None', finish_reasons)
    invocation = extra.get('invocation')
    failed = (
        error_type is not None
        or 'error' in (finish_reasons or [])
        or (isinstance(invocation, dict) and cast('dict[str, Any]', invocation).get('status') == 'error')
    )
    return error_type, finish_reasons, failed


def _metrics_from_usage(usage: ResponseUsage | None, unset: frozenset[str]) -> AtifMetrics | None:
    """Step metrics from a Response's usage; counts named in `unset` were written as 0 for an unset ATIF value."""
    if usage is None:
        return None

    def count(name: str, value: int) -> int | None:
        return None if name in unset else value

    extra = {
        'reasoning_tokens': count('reasoning_tokens', usage.output_tokens_details.reasoning_tokens),
        'total_tokens': count('total_tokens', usage.total_tokens),
    }
    metrics = AtifMetrics(
        prompt_tokens=count('prompt_tokens', usage.input_tokens),
        completion_tokens=count('completion_tokens', usage.output_tokens),
        cached_tokens=count('cached_tokens', usage.input_tokens_details.cached_tokens),
        extra={key: value for key, value in extra.items() if value is not None} or None,
    )
    return None if metrics == AtifMetrics() else metrics


# ATIF -> Responses


def atif_to_responses(traj: AtifTrajectory) -> ResponsesConversation:
    """Render an ATIF trajectory as Responses items plus one `Response` per agent step with output items.

    Each `Response.output` holds the output items of its step (assistant message, reasoning, function calls).
    A step containing only tool results has no model output and therefore gets no `Response`.

    A step without model, metrics or timestamp gets a placeholder (`model=''`, `usage=None`, `created_at=0.0`),
    which `responses_to_atif` reads back as absent. Responses usage has no unset token count, so an unset ATIF
    count is written as 0 and named in `Response.metadata['atif_unset_usage']`, which reads back as unset.

    A step's outcome (`extra.error_type`, `extra.finish_reasons`, an `extra.invocation` status of `error`) sets
    `status`, `error` and `incomplete_details`: a failed call is `failed`, a `length` or `content_filter` finish
    is `incomplete`. What those closed enums cannot hold is kept in `Response.metadata['atif_error_type']` and
    `['atif_finish_reasons']`, which win on the way back. An explicit `extra.status`, `extra.error` or
    `extra.incomplete_details` overrides the derived value.

    Lost: embedded and referenced subagent trajectories (warned), results with no `source_call_id`
    (warned), metrics `cost_usd`, `prompt_token_ids`, `completion_token_ids` and `logprobs` (warned),
    `reasoning_effort`, `llm_call_count` (read back as 1 for a step with a `Response` that is not a
    placeholder), `extra.response_id` of a step with no model, metrics or timestamp, non-text parts in agent
    messages, and audio or file-path images in user messages (rendered as `[audio: path]` / `[image: path]`
    text markers).
    """
    seed = traj.session_id or traj.trajectory_id or content_seed(traj.model_dump(mode='json'))
    for sub in traj.subagent_trajectories or []:
        logger.warning('subagent {} not representable in Responses', sub.trajectory_id)
    items: list[dict[str, Any]] = []
    responses: list[Response] = []
    unmapped: dict[str, int] = {}
    for step in traj.steps:
        if step.source == 'agent':
            step_items = _agent_items(step, seed)
            items.extend(step_items)
            output = [output_item(i) for i in step_items if is_output_item(i)]
            if output:
                responses.append(_response(step, seed, output))
            for name in _UNMAPPED_METRICS:
                if step.metrics is not None and getattr(step.metrics, name) is not None:
                    unmapped[name] = unmapped.get(name, 0) + 1
        else:
            extra = step.extra or {}
            context = extra.get('context_management')
            compactions = extra.get(COMPACTION_EXTRA_KEY)
            if isinstance(context, dict) and context.get('type') == 'compaction':
                restored = (
                    [item for item in compactions if isinstance(item, dict) and item.get('type') == 'compaction']
                    if isinstance(compactions, list)
                    else []
                )
                if isinstance(compactions, list) and len(restored) == len(compactions) and restored:
                    items.extend(restored)
                    continue
                logger.warning(
                    'Compaction step {} has no Responses compaction item; rendering its message.', step.step_id
                )
            role = 'developer' if (step.extra or {}).get('original_role') == 'developer' else step.source
            content = step.message if isinstance(step.message, str) else _atif_parts(step.message)
            items.extend(messages_to_responses_input([Message(role=role, content=content)]))
    if unmapped:
        logger.warning('Responses usage has no slot for step metrics {} (steps per metric); dropping them.', unmapped)
    for item in items:
        if 'role' in item:
            item.setdefault('type', 'message')
    return ResponsesConversation(items=items, responses=responses or None)


def _atif_parts(parts: list[AtifContentPart]) -> list[ContentPart]:
    out: list[ContentPart] = []
    for part in parts:
        if part.type == 'text':
            out.append(InputTextContent(type='input_text', text=part.text or ''))
            continue
        path = part.source.path if part.source is not None else ''
        if part.type == 'image' and path.startswith(('http://', 'https://', 'data:')):
            out.append(InputImageContent(type='input_image', image_url=path))
            continue
        out.append(InputTextContent(type='input_text', text=media_marker(part, 'Responses input')))
    return out


def _agent_items(step: AtifStep, seed: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if step.reasoning_content:
        items.append({
            'type': 'reasoning',
            'id': 'rs_' + stable_hex(seed, str(step.step_id), length=24),
            'summary': [{'type': 'summary_text', 'text': step.reasoning_content}],
        })
    fc_item_ids = (step.extra or {}).get('fc_item_ids') or {}
    if not isinstance(fc_item_ids, dict):
        _warn_foreign('fc_item_ids', step.step_id, fc_item_ids)
        fc_item_ids = {}
    fc_item_ids = cast('dict[str, Any]', fc_item_ids)
    calls = [
        StrategyToolCall(
            id=call.tool_call_id,
            function=FunctionCall(
                name=call.function_name,
                arguments=(
                    call.extra[RAW_ARGUMENTS_EXTRA_KEY]
                    if isinstance(call.extra, dict) and isinstance(call.extra.get(RAW_ARGUMENTS_EXTRA_KEY), str)
                    else json_arguments_text(call.arguments)
                ),
            ),
            item_id=item_id if isinstance(item_id := fc_item_ids.get(call.tool_call_id), str) else None,
        )
        for call in step.tool_calls or []
    ]
    names = {call.tool_call_id: call.function_name for call in step.tool_calls or []}
    content = step.message if isinstance(step.message, str) else _atif_parts(step.message)
    extra = step.extra or {}
    has_non_message_output = bool(step.reasoning_content or extra.get(_UNMAPPED_OUTPUTS_KEY))
    has_tool_results = bool(extra.get(_UNMAPPED_RESULTS_KEY)) or bool(
        step.observation is not None and step.observation.results
    )
    messages = (
        [Message(role='assistant', content=content or None, tool_calls=calls or None)]
        if content or calls or not (has_non_message_output or has_tool_results)
        else []
    )
    for result in step.observation.results if step.observation else []:
        for ref in result.subagent_trajectory_ref or []:
            logger.warning('subagent {} not representable in Responses', ref.trajectory_id or ref.trajectory_path)
        if result.source_call_id is None:
            logger.warning(
                'Dropping an observation result in step {} with no source_call_id: a function_call_output '
                'needs a call_id.',
                step.step_id,
            )
            continue
        if isinstance(result.content, list) and all(part.type == 'text' for part in result.content):
            result_content = atif_content_text(result.content, 'Responses')
        elif isinstance(result.content, list):
            result_content = _atif_parts(result.content)
        else:
            result_content = result.content
        messages.append(
            Message(
                role='tool',
                tool_call_id=result.source_call_id,
                name=names.get(result.source_call_id),
                content=result_content or '',
            )
        )
    items.extend(messages_to_responses_input(messages))
    outputs: list[dict[str, Any]] = []
    unmapped = (step.extra or {}).get(_UNMAPPED_OUTPUTS_KEY)
    if isinstance(unmapped, list):
        outputs = [item for item in unmapped if isinstance(item, dict) and is_output_item(item)]
        if len(outputs) != len(unmapped):
            logger.warning('Step {} has malformed Responses output items; dropping them.', step.step_id)
        insertion = next(
            (index for index, item in enumerate(items) if item.get('type') == 'function_call_output'), len(items)
        )
        items[insertion:insertion] = outputs
    results: list[dict[str, Any]] = []
    raw_results = (step.extra or {}).get(_UNMAPPED_RESULTS_KEY)
    if isinstance(raw_results, list):
        results = [
            item for item in raw_results if isinstance(item, dict) and item.get('type') == 'custom_tool_call_output'
        ]
        if len(results) != len(raw_results):
            logger.warning('Step {} has malformed Responses custom tool results; dropping them.', step.step_id)
    _order_result_items(items, results, (step.extra or {}).get(_RESULT_ORDER_KEY))
    return items


def _order_result_items(items: list[dict[str, Any]], custom_results: list[dict[str, Any]], order: Any) -> None:
    ordinary = [item for item in items if item.get('type') == 'function_call_output']
    if not isinstance(order, list):
        items.extend(custom_results)
        return
    expected = {('function', index) for index in range(len(ordinary))} | {
        ('custom', index) for index in range(len(custom_results))
    }
    observed: list[tuple[str, int]] = []
    for entry in order:
        if (
            not isinstance(entry, dict)
            or entry.get('type') not in ('function', 'custom')
            or not isinstance(entry.get('index'), int)
            or isinstance(entry.get('index'), bool)
        ):
            observed = []
            break
        observed.append((entry['type'], entry['index']))
    if len(observed) != len(expected) or set(observed) != expected:
        logger.warning('Malformed Responses result order metadata; restoring function results before custom results.')
        items[:] = [item for item in items if item.get('type') != 'function_call_output']
        insertion = next(
            (index for index, item in enumerate(items) if item.get('type') == 'custom_tool_call_output'), len(items)
        )
        items[insertion:insertion] = [*ordinary, *custom_results]
        return
    combined: list[dict[str, Any]] = []
    for entry in order:
        if not isinstance(entry, dict) or not isinstance(entry.get('index'), int):
            continue
        index = entry['index']
        if entry.get('type') == 'function' and 0 <= index < len(ordinary):
            combined.append(ordinary[index])
        elif entry.get('type') == 'custom' and 0 <= index < len(custom_results):
            combined.append(custom_results[index])
    if len(combined) != len(ordinary) + len(custom_results):
        combined = [*ordinary, *custom_results]
    items[:] = [item for item in items if item.get('type') != 'function_call_output']
    items.extend(combined)


def _response(step: AtifStep, seed: str, output: list[ResponseOutputItem]) -> Response:
    extra = step.extra or {}
    usage: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None
    if step.metrics is not None:
        metrics = step.metrics
        counts = {
            'prompt_tokens': metrics.prompt_tokens,
            'completion_tokens': metrics.completion_tokens,
            'cached_tokens': metrics.cached_tokens,
            'reasoning_tokens': _extra_int(metrics.extra, 'reasoning_tokens', step.step_id),
            'total_tokens': _extra_int(metrics.extra, 'total_tokens', step.step_id),
        }
        prompt = counts['prompt_tokens'] or 0
        completion = counts['completion_tokens'] or 0
        usage = {
            'input_tokens': prompt,
            'output_tokens': completion,
            'total_tokens': counts['total_tokens'] if counts['total_tokens'] is not None else prompt + completion,
            'input_tokens_details': {'cached_tokens': counts['cached_tokens'] or 0},
            'output_tokens_details': {'reasoning_tokens': counts['reasoning_tokens'] or 0},
        }
        unset = [name for name, value in counts.items() if value is None]
        metadata = {UNSET_USAGE_KEY: ','.join(unset)} if unset else None
    response_id = extra.get('response_id')
    if response_id is not None and not isinstance(response_id, str):
        _warn_foreign('response_id', step.step_id, response_id)
        response_id = None
    status = extra.get('status')
    if status is not None and not (isinstance(status, str) and status in _RESPONSE_STATUSES):
        _warn_foreign('status', step.step_id, status)
        status = None
    tools = extra.get(_RESPONSE_TOOLS_KEY, [])
    if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
        _warn_foreign(_RESPONSE_TOOLS_KEY, step.step_id, tools)
        tools = []
    error_type, finish_reasons, failed = _step_outcome(step)
    derived = _outcome_fields(error_type, finish_reasons, failed=failed)
    error = _extra_model(extra, 'error', ResponseError, step.step_id) or derived['error']
    incomplete = (
        _extra_model(extra, 'incomplete_details', IncompleteDetails, step.step_id) or (derived['incomplete_details'])
    )
    read_back = _read_outcome(error['code'] if error else None, incomplete['reason'] if incomplete else None, None)
    if read_back[0] != error_type:
        metadata = {**(metadata or {}), ERROR_TYPE_KEY: error_type or ''}
    if read_back[1] != finish_reasons:
        metadata = {**(metadata or {}), FINISH_REASONS_KEY: json.dumps(finish_reasons)}
    return Response.model_validate({
        'id': response_id or 'resp_' + stable_hex(seed, 'response', str(step.step_id), length=24),
        'created_at': _epoch(step.timestamp),
        'model': step.model_name or '',
        'object': 'response',
        'output': output,
        'parallel_tool_calls': False,
        'tool_choice': 'auto',
        'tools': tools,
        'status': status or derived['status'],
        'error': error,
        'incomplete_details': incomplete,
        'usage': usage,
        'metadata': metadata,
    })


def _warn_foreign(key: str, step_id: int, value: object) -> None:
    logger.warning(
        'Step {} extra.{} = {!r} is not a Responses value; dropping it from the Response.', step_id, key, value
    )


def _extra_int(extra: dict[str, Any] | None, key: str, step_id: int) -> int | None:
    value = (extra or {}).get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    _warn_foreign(key, step_id, value)
    return None


def _extra_model(extra: dict[str, Any], key: str, model: type[BaseModel], step_id: int) -> dict[str, Any] | None:
    value = extra.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            model.model_validate(value)
        except ValidationError:
            pass
        else:
            return cast('dict[str, Any]', value)
    _warn_foreign(key, step_id, value)
    return None


def _epoch(timestamp: str | None) -> float:
    if timestamp is None:
        return 0.0
    moment = parse_iso(timestamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()
