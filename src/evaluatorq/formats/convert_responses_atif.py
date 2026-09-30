"""Convert between Responses transcripts and ATIF trajectories.

`items` are authoritative: steps are cut from them. `responses` only enrich agent steps with model,
usage, timing and finish status, and only when there is exactly one per agent step. `atif_to_responses`
emits one `Response` per agent step, but consecutive agent steps with no tool result between them merge
into one step when the items are read back, so their `Response` list no longer lines up and is ignored.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from loguru import logger
from openai.types.responses import Response, ResponseError, ResponseStatus
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
from evaluatorq.formats._shared import arguments_text, final_metrics, media_marker, tool_arguments
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
from evaluatorq.formats.responses import ResponsesConversation
from evaluatorq.openresponses.input_items import messages_to_responses_input

if TYPE_CHECKING:
    from openai.types.responses import ResponseUsage

_TEXT_PART_TYPES = frozenset({'input_text', 'output_text', 'text'})
_OUTPUT_ITEM_TYPES = frozenset({'message', 'reasoning', 'function_call'})
_IMAGE_MEDIA_TYPES: dict[str, Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']] = {
    'image/jpeg': 'image/jpeg',
    'image/png': 'image/png',
    'image/gif': 'image/gif',
    'image/webp': 'image/webp',
}
_RESPONSE_STATUSES: frozenset[str] = frozenset(get_args(ResponseStatus))
_NO_STEPS = 'ResponsesConversation has no items; ATIF needs at least one step'


@dataclass
class _Draft:
    """One ATIF step being assembled from consecutive Responses items."""

    source: Literal['system', 'user', 'agent']
    texts: list[str] = field(default_factory=list)
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    seen_result: bool = False
    item: dict[str, Any] | None = None  # the message item of a user/system step


# Responses -> ATIF


def responses_to_atif(
    conv: ResponsesConversation,
    *,
    agent_name: str = 'unknown',
    agent_version: str = 'unknown',
    session_id: str | None = None,
) -> AtifTrajectory:
    """Build an ATIF trajectory from Responses items; see `ResponsesConversation.to_atif` for the mapping."""
    drafts = _segment(conv.items)
    if not drafts:
        raise ValueError(_NO_STEPS)
    agent_count = sum(1 for d in drafts if d.source == 'agent')
    responses: list[Response | None] = [None] * agent_count
    if conv.responses is not None:
        if len(conv.responses) == agent_count:
            responses = list(conv.responses)
        else:
            logger.warning(
                'Got {} responses for {} agent steps; ignoring all responses (items stay authoritative).',
                len(conv.responses),
                agent_count,
            )
    steps: list[AtifStep] = []
    agent_index = 0
    for index, draft in enumerate(drafts):
        if draft.source == 'agent':
            steps.append(_agent_step(draft, index + 1, responses[agent_index]))
            agent_index += 1
        else:
            steps.append(_user_step(draft, index + 1))
    return AtifTrajectory(
        schema_version='ATIF-v1.7',
        session_id=session_id or stable_hex(content_seed(conv.items), length=16),
        agent=AtifAgent(name=agent_name, version=agent_version),
        steps=steps,
        final_metrics=final_metrics(steps),
    )


def _segment(items: list[dict[str, Any]]) -> list[_Draft]:
    """Cut Responses items into step drafts (user/system messages, and agent turns closed by tool results)."""
    drafts: list[_Draft] = []

    def current_agent() -> _Draft:
        last = drafts[-1] if drafts else None
        if last is None or last.source != 'agent' or last.seen_result:
            last = _Draft(source='agent')
            drafts.append(last)
        return last

    for item in items:
        item_type = item.get('type')
        if item_type is None and 'role' in item:
            item_type = 'message'
        role = item.get('role')
        if item_type == 'message' and role in ('user', 'system', 'developer'):
            drafts.append(_Draft(source='user' if role == 'user' else 'system', item=item))
        elif item_type == 'message' and role == 'assistant':
            current_agent().texts.append(_assistant_text(item.get('content')))
        elif item_type == 'message':
            logger.warning('Skipping Responses message with role {!r}: ATIF has no such step source.', role)
        elif item_type == 'reasoning':
            current_agent().reasoning.append(item)
        elif item_type == 'function_call':
            if isinstance(item.get('call_id'), str) and item['call_id']:
                current_agent().calls.append(item)
            else:
                logger.warning('Responses function_call {!r} has no call_id; skipping it.', item.get('name'))
        elif item_type == 'function_call_output':
            _attach_output(item, drafts)
        else:
            logger.warning('Skipping Responses item of type {!r}: ATIF has no equivalent.', item_type)
    return drafts


def _attach_output(item: dict[str, Any], drafts: list[_Draft]) -> None:
    call_id = item.get('call_id')
    output = item.get('output')
    content = output if isinstance(output, str) else tool_result_to_text(output)
    for draft in reversed(drafts):
        if draft.source == 'agent' and any(c.get('call_id') == call_id for c in draft.calls):
            draft.results.append({'source_call_id': call_id, 'content': content})
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
    last.seen_result = True


def _assistant_text(content: Any) -> str:
    """Flatten an assistant message item's content to text."""
    if content is None or isinstance(content, str):
        return content_to_text(content)
    if not isinstance(content, list):
        logger.warning('Assistant message content is {}; dropping it.', type(content).__name__)
        return ''
    parts: list[ContentPart] = []
    for part in content:
        part_type = part.get('type') if isinstance(part, dict) else None
        if part_type in _TEXT_PART_TYPES:
            parts.append(InputTextContent(type='input_text', text=str(part.get('text') or '')))
        elif part_type == 'refusal':
            parts.append(InputTextContent(type='input_text', text=str(part.get('refusal') or '')))
        else:
            logger.warning('Dropping assistant content part of type {!r}: ATIF agent messages are text.', part_type)
    return content_to_text(parts)


def _user_step(draft: _Draft, step_id: int) -> AtifStep:
    item = draft.item or {}
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
        return '\n'.join(p.text or '' for p in parts)
    return parts


def _user_part(part: Any) -> AtifContentPart:
    part_type = part.get('type') if isinstance(part, dict) else type(part).__name__
    if isinstance(part, dict) and part_type in _TEXT_PART_TYPES:
        return AtifContentPart(type='text', text=str(part.get('text') or ''))
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


def _agent_step(draft: _Draft, step_id: int, response: Response | None) -> AtifStep:
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
    results = [AtifObservationResult(**result) for result in draft.results]
    fields: dict[str, Any] = {}
    if response is not None:
        extra.update(_response_extra(response))
        fields = {
            'model_name': response.model or None,
            'timestamp': (
                datetime.fromtimestamp(response.created_at, tz=timezone.utc).isoformat()
                if response.created_at
                else None
            ),
            'metrics': _metrics_from_usage(response.usage),
            'llm_call_count': 1,
        }
    return AtifStep(
        step_id=step_id,
        source='agent',
        message='\n'.join(draft.texts),
        reasoning_content='\n'.join(reasoning_texts) or None,
        tool_calls=tool_calls or None,
        observation=AtifObservation(results=results) if results else None,
        extra=extra or None,
        **fields,
    )


def _tool_call(call: dict[str, Any]) -> AtifToolCall:
    return AtifToolCall(
        tool_call_id=call['call_id'],
        function_name=str(call.get('name') or ''),
        arguments=tool_arguments(call.get('arguments'), call.get('name')),
    )


def _response_extra(response: Response) -> dict[str, Any]:
    extra: dict[str, Any] = {'response_id': response.id}
    if response.status is not None and response.status != 'completed':
        extra['status'] = response.status
    if response.error is not None:
        extra['error'] = response.error.model_dump(mode='json')
    if response.incomplete_details is not None:
        extra['incomplete_details'] = response.incomplete_details.model_dump(mode='json')
    return extra


def _metrics_from_usage(usage: ResponseUsage | None) -> AtifMetrics | None:
    if usage is None:
        return None
    return AtifMetrics(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        extra={'reasoning_tokens': usage.output_tokens_details.reasoning_tokens, 'total_tokens': usage.total_tokens},
    )


# ATIF -> Responses


def atif_to_responses(traj: AtifTrajectory) -> ResponsesConversation:
    """Render an ATIF trajectory as Responses items plus exactly one `Response` per agent step.

    A step without model, metrics or timestamp gets a placeholder (`model=''`, `usage=None`, `created_at=0.0`),
    which `responses_to_atif` reads back as absent.

    Lost: embedded and referenced subagent trajectories (warned), results with no `source_call_id`
    (warned), non-text parts in agent messages, and audio or file-path images in user messages
    (rendered as `[audio: path]` / `[image: path]` text markers).
    """
    seed = traj.session_id or traj.trajectory_id or content_seed(traj.model_dump(mode='json'))
    for sub in traj.subagent_trajectories or []:
        logger.warning('subagent {} not representable in Responses', sub.trajectory_id)
    items: list[dict[str, Any]] = []
    responses: list[Response] = []
    for step in traj.steps:
        if step.source == 'agent':
            items.extend(_agent_items(step, seed))
            responses.append(_response(step, seed))
        else:
            role = 'developer' if (step.extra or {}).get('original_role') == 'developer' else step.source
            content = step.message if isinstance(step.message, str) else _atif_parts(step.message)
            items.extend(messages_to_responses_input([Message(role=role, content=content)]))
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
            function=FunctionCall(name=call.function_name, arguments=arguments_text(call.arguments)),
            item_id=item_id if isinstance(item_id := fc_item_ids.get(call.tool_call_id), str) else None,
        )
        for call in step.tool_calls or []
    ]
    names = {call.tool_call_id: call.function_name for call in step.tool_calls or []}
    content = step.message if isinstance(step.message, str) else _atif_parts(step.message)
    messages = [Message(role='assistant', content=content or None, tool_calls=calls or None)]
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
        result_content = (
            result.content if isinstance(result.content, str) or result.content is None else _atif_parts(result.content)
        )
        messages.append(
            Message(
                role='tool',
                tool_call_id=result.source_call_id,
                name=names.get(result.source_call_id),
                content=result_content or '',
            )
        )
    items.extend(messages_to_responses_input(messages))
    return items


def _response(step: AtifStep, seed: str) -> Response:
    extra = step.extra or {}
    usage: dict[str, Any] | None = None
    if step.metrics is not None:
        metrics = step.metrics
        prompt = metrics.prompt_tokens or 0
        completion = metrics.completion_tokens or 0
        usage = {
            'input_tokens': prompt,
            'output_tokens': completion,
            'total_tokens': _extra_int(metrics.extra, 'total_tokens', step.step_id) or prompt + completion,
            'input_tokens_details': {'cached_tokens': metrics.cached_tokens or 0},
            'output_tokens_details': {'reasoning_tokens': _extra_int(metrics.extra, 'reasoning_tokens', step.step_id)},
        }
    response_id = extra.get('response_id')
    if response_id is not None and not isinstance(response_id, str):
        _warn_foreign('response_id', step.step_id, response_id)
        response_id = None
    status = extra.get('status')
    if status is not None and not (isinstance(status, str) and status in _RESPONSE_STATUSES):
        _warn_foreign('status', step.step_id, status)
        status = None
    return Response.model_validate({
        'id': response_id or 'resp_' + stable_hex(seed, 'response', str(step.step_id), length=24),
        'created_at': _epoch(step.timestamp),
        'model': step.model_name or '',
        'object': 'response',
        'output': [],
        'parallel_tool_calls': False,
        'tool_choice': 'auto',
        'tools': [],
        'status': status or 'completed',
        'error': _extra_model(extra, 'error', ResponseError, step.step_id),
        'incomplete_details': _extra_model(extra, 'incomplete_details', IncompleteDetails, step.step_id),
        'usage': usage,
    })


def _warn_foreign(key: str, step_id: int, value: object) -> None:
    logger.warning(
        'Step {} extra.{} = {!r} is not a Responses value; dropping it from the Response.', step_id, key, value
    )


def _extra_int(extra: dict[str, Any] | None, key: str, step_id: int) -> int:
    value = (extra or {}).get(key)
    if value is None:
        return 0
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    _warn_foreign(key, step_id, value)
    return 0


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
    moment = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()
