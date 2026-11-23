"""A fake of the Orq ``/v3/traces`` routes for `fetch_traces` tests, in the shapes the real API serves.

Fixtures hold each span as one dict: its summary fields, its ``attributes``, and
optionally a ``conversation`` list of Responses items. The fake splits them the
way the API does. The span list returns summaries without content, the span
detail returns ``{'span': {'summary': ..., 'attributes': ...}}`` and nothing
else, and the conversation route returns ``{'trace_id', 'span_id', 'items'}``
for the latest span that carries a ``conversation`` (or the requested one).
"""

from __future__ import annotations

from typing import Any

import httpx

SEARCH_PATH = '/v3/traces/search'

_CONTENT_KEYS = frozenset({'attributes', 'conversation'})


def search_page(trace_ids: list[str], *, next_page_token: str = '') -> httpx.Response:
    """One ``/v3/traces/search`` page listing ``trace_ids``."""
    return httpx.Response(
        200,
        json={
            'object': 'list',
            'data': [{'trace_id': trace_id} for trace_id in trace_ids],
            'has_more': bool(next_page_token),
            'next_page_token': next_page_token,
        },
    )


def message_item(role: str, text: str) -> dict[str, Any]:
    """One Responses ``message`` item as the conversation route serves it."""
    part_type = 'output_text' if role == 'assistant' else 'input_text'
    return {'type': 'message', 'role': role, 'content': [{'type': part_type, 'text': text}]}


def _summary(span: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in span.items() if key not in _CONTENT_KEYS}


def _conversation(request: httpx.Request, trace_id: str, spans: Any) -> httpx.Response:
    if not isinstance(spans, list):
        return httpx.Response(200, json=spans)
    spans = [span for span in spans if isinstance(span, dict)]
    requested = request.url.params.get('span_id')
    if requested:
        span = next((span for span in spans if span.get('span_id') == requested), None)
        if span is None:
            return httpx.Response(404, json={'message': 'trace span not found'})
        candidates = [span] if span.get('conversation') else []
    else:
        candidates = [span for span in spans if span.get('conversation')]
    if not candidates:
        return httpx.Response(200, json={'trace_id': trace_id, 'span_id': '', 'items': []})
    span = max(enumerate(candidates), key=lambda item: (str(item[1].get('started_at') or ''), item[0]))[1]
    return httpx.Response(
        200, json={'trace_id': trace_id, 'span_id': span['span_id'], 'items': span['conversation']}
    )


def serve_spans(request: httpx.Request, spans_by_trace: dict[str, Any]) -> httpx.Response | None:
    """Answer the span list, span detail and conversation routes, or return None for any other path.

    A trace whose value is not a list is served as the raw span-list and
    conversation body, so a test can hand back a malformed payload.
    """
    parts = request.url.path.strip('/').split('/')
    if parts[:2] != ['v3', 'traces'] or len(parts) not in (4, 5):
        return None
    spans = spans_by_trace.get(parts[2], [])
    if parts[3] == 'conversation' and len(parts) == 4:
        if parts[2] not in spans_by_trace:
            return httpx.Response(404, json={'message': 'trace not found'})
        return _conversation(request, parts[2], spans)
    if parts[3] != 'spans':
        return None
    if len(parts) == 4:
        if not isinstance(spans, list):
            return httpx.Response(200, json=spans)
        summaries = [_summary(span) if isinstance(span, dict) else span for span in spans]
        return httpx.Response(200, json={'object': 'list', 'data': summaries, 'has_more': False, 'next_page_token': ''})
    for span in spans if isinstance(spans, list) else []:
        if isinstance(span, dict) and span.get('span_id') == parts[4]:
            return httpx.Response(200, json={'span': {'summary': _summary(span), 'attributes': span.get('attributes', {})}})
    return httpx.Response(404, json={'message': 'trace span not found'})
