"""A fake of the Orq ``/v3/traces`` span routes for `fetch_traces` tests.

Fixtures hold each span as one full dict. The real API splits it: the list
route returns summaries without content, and a per-span detail route returns
``{'span': {'summary': ..., 'attributes': ...}}``. Serving fixtures through
this split is what proves the importer reads both routes.
"""

from __future__ import annotations

from typing import Any

import httpx

SEARCH_PATH = '/v3/traces/search'

_CONTENT_KEYS = frozenset({'attributes', 'events', 'input', 'output'})


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


def serve_spans(request: httpx.Request, spans_by_trace: dict[str, Any]) -> httpx.Response | None:
    """Answer the span list and span detail routes, or return None for any other path.

    A trace whose value is not a list is served as the raw list body, so a test
    can hand back a malformed payload. A span without a ``span_id`` is listed
    whole, because the importer has no detail route to reach it by.
    """
    parts = request.url.path.strip('/').split('/')
    if parts[:2] != ['v3', 'traces'] or len(parts) not in (4, 5) or parts[3] != 'spans':
        return None
    spans = spans_by_trace.get(parts[2], [])
    if len(parts) == 4:
        if not isinstance(spans, list):
            return httpx.Response(200, json=spans)
        summaries = [
            {key: value for key, value in span.items() if key not in _CONTENT_KEYS}
            if isinstance(span, dict) and span.get('span_id')
            else span
            for span in spans
        ]
        return httpx.Response(200, json={'object': 'list', 'data': summaries, 'has_more': False, 'next_page_token': ''})
    for span in spans if isinstance(spans, list) else []:
        if isinstance(span, dict) and span.get('span_id') == parts[4]:
            summary = {key: value for key, value in span.items() if key not in _CONTENT_KEYS}
            content = {key: value for key, value in span.items() if key in _CONTENT_KEYS}
            return httpx.Response(200, json={'span': {'summary': summary, **content}})
    return httpx.Response(404, json={'message': 'span not found'})
