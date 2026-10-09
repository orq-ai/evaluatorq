"""Character-bounded trace inputs for classification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.model_input import (
    JEV_STATE_CHARS,
    cap_text,
    effective_trace_input_chars,
    is_jev_model,
    jev_state,
)
from evaluatorq.common.trace_document import prompt_messages
from evaluatorq.contracts import tool_result_to_text
from evaluatorq.trace_finder.settings import MAX_TRACE_INPUT_CHARS, MIN_TRACE_INPUT_CHARS

from .models import TraceProjection, TraceRecord

if TYPE_CHECKING:
    from evaluatorq.common.trace_document import TraceDocument

ERROR_STATUSES = frozenset({'error', 'failed', 'failure', 'cancelled', 'canceled'})
SUCCESS_STATUSES = frozenset({'completed', 'success', 'succeeded', 'ok'})
_TOOL_RESULT_CATEGORIES = (
    ('not_found', ('not found', 'not_found', 'does not exist', 'no such record')),
    ('permission_denied', ('permission denied', 'forbidden', 'unauthorized', 'access denied')),
    ('rate_limited', ('rate limit', 'rate_limit', 'too many requests')),
    ('timeout', ('timed out', 'timeout', 'deadline exceeded')),
    ('unavailable', ('unavailable', 'connection refused', 'connection reset')),
    ('invalid_request', ('invalid request', 'invalid argument', 'bad request')),
)


def serialize_projection(payload: dict[str, Any]) -> str:
    """Return the stable representation that classifier receives and the drawer displays."""

    return _canonical_json(payload)


def estimate_tokens(serialized: str) -> int:
    """Estimate tokens at four characters per token; this is not a model tokenizer count."""
    return (len(serialized) + 3) // 4


def project_trace(
    trace: TraceRecord | TraceDocument,
    *,
    model: str = 'typesafe/jev-latest',
    trace_input_chars: int | None = None,
) -> TraceProjection:
    """Render complete readable source text within the persisted classifier-input cap."""
    character_cap = effective_trace_input_chars() if trace_input_chars is None else trace_input_chars
    if not MIN_TRACE_INPUT_CHARS <= character_cap <= MAX_TRACE_INPUT_CHARS:
        raise ValueError(
            f'trace_input_chars must be between {MIN_TRACE_INPUT_CHARS} and {MAX_TRACE_INPUT_CHARS} characters'
        )
    if is_jev_model(model):
        source = prompt_messages(trace)
        empty_state = {'messages': []}
        state_overhead = len(serialize_projection({'trace_status': trace.status, **empty_state})) - len(
            serialize_projection(empty_state)
        )
        state_cap = min(JEV_STATE_CHARS, character_cap - state_overhead)
        payload = {'trace_status': trace.status, **jev_state(source, global_char_cap=state_cap)}
        serialized = serialize_projection(payload)
        represented = {entry['index'] for entry in payload.get('messages', []) if isinstance(entry, dict)}
        for entry in payload.get('messages', []):
            if not isinstance(entry, dict):
                continue
            represented.update(
                result['index']
                for result in entry.get('results', [])
                if isinstance(result, dict) and isinstance(result.get('index'), int)
            )
        return TraceProjection(
            payload=payload,
            serialized=serialized,
            estimated_tokens=estimate_tokens(serialized),
            omitted_messages=max(0, len(source) - len(represented)),
            omitted_bytes=max(
                0,
                sum(_source_message_bytes(message) for message in source) - len(serialized.encode('utf-8')),
            ),
        )

    return _project_full_conversation(trace, character_cap)


def _project_full_conversation(trace: TraceRecord | TraceDocument, character_cap: int) -> TraceProjection:
    """Render the complete readable conversation for non-Jev classifiers under the saved cap."""
    from evaluatorq.insights.transcript import full_conversation_view

    conversation = full_conversation_view(trace, budget=character_cap)
    original_conversation = conversation
    payload = {'trace_status': trace.status, 'conversation': conversation}
    serialized = serialize_projection(payload)
    if len(serialized) > character_cap:
        empty_payload = {'trace_status': trace.status, 'conversation': ''}
        empty_serialized = serialize_projection(empty_payload)
        if len(empty_serialized) > character_cap:
            raise ValueError('trace_input_chars is too small for required classifier state metadata')
        best_conversation = ''
        best_payload = empty_payload
        best_serialized = empty_serialized
        low = len('[... 0 chars left out ...]')
        high = len(conversation)
        while low <= high:
            text_cap = (low + high) // 2
            candidate = cap_text(conversation, text_cap)[0]
            candidate_payload = {'trace_status': trace.status, 'conversation': candidate}
            candidate_serialized = serialize_projection(candidate_payload)
            if len(candidate_serialized) <= character_cap:
                best_conversation = candidate
                best_payload = candidate_payload
                best_serialized = candidate_serialized
                low = text_cap + 1
            else:
                high = text_cap - 1
        conversation = best_conversation
        payload = best_payload
        serialized = best_serialized
    source = prompt_messages(trace)
    source_bytes = sum(_source_message_bytes(message) for message in source)
    if conversation != original_conversation:
        logger.warning(
            'trace {} conversation was truncated to fit the {} character model input limit',
            trace.trace_id,
            character_cap,
        )
    return TraceProjection(
        payload=payload,
        serialized=serialized,
        estimated_tokens=estimate_tokens(serialized),
        omitted_messages=0,
        omitted_bytes=max(0, source_bytes - len(serialized.encode('utf-8'))),
    )


def _matching_tool_result(call_id: Any, results: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    """Do not pair an idless call with an unrelated idless result."""
    if not isinstance(call_id, str) or not call_id:
        return None
    return next((item for item in results if item.get('tool_call_id') == call_id), None)


def _tool_result_category(call_id: Any, results: tuple[dict[str, Any], ...]) -> str | None:
    """Return a fixed diagnostic label without projecting provider result bytes."""
    result = _matching_tool_result(call_id, results)
    if result is None or _tool_call_status(call_id, results) != 'error':
        return None
    text = tool_result_to_text(result.get('content')).casefold()
    category = next(
        (name for name, markers in _TOOL_RESULT_CATEGORIES if any(marker in text for marker in markers)),
        None,
    )
    return category or 'provider_error'


def _tool_call_status(call_id: Any, results: tuple[dict[str, Any], ...]) -> str:
    result = _matching_tool_result(call_id, results)
    if result is None:
        return 'pending'
    metadata = result.get('trace_finder_metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    nested = metadata.get('tool_result')
    sources = [source for source in (nested, result) if isinstance(source, dict)]
    for source in sources:
        status = source.get('status')
        if isinstance(status, str) and status.casefold() in ERROR_STATUSES:
            return 'error'
        if source.get('is_error') is True or source.get('error') not in (None, False, ''):
            return 'error'
    for source in sources:
        status = source.get('status')
        if isinstance(status, str) and status.casefold() in SUCCESS_STATUSES:
            return 'completed'
        if source.get('is_error') is False:
            return 'completed'
    return _content_tool_call_status(tool_result_to_text(result.get('content')).lstrip()) or 'completed'


def _content_tool_call_status(content: str) -> str | None:
    try:
        decoded = json.loads(content)
    except (TypeError, ValueError):
        decoded = None
    if isinstance(decoded, dict):
        status = decoded.get('status')
        if isinstance(status, str) and status.casefold() in ERROR_STATUSES:
            return 'error'
        if decoded.get('error') not in (None, False, ''):
            return 'error'
        if isinstance(status, str) and status.casefold() in SUCCESS_STATUSES:
            return 'completed'
        if decoded.get('is_error') is False:
            return 'completed'
    return 'error' if content.casefold().startswith('error:') else None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _source_message_bytes(message: dict[str, Any]) -> int:
    return len(serialize_projection(message).encode('utf-8'))
