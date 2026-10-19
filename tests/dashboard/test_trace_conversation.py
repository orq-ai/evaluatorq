from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evaluatorq.dashboard.trace_finder.routes import span_tree_fragment
from evaluatorq.dashboard.trace_finder.views import (
    drawer,
    missing_trace_drawer,
    trace_conversation,
    unavailable_conversation,
)
from evaluatorq.trace_finder.models import TraceDetail, TraceRecord


def _record() -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 10, 1, tzinfo=timezone.utc),
        messages=(
            {'role': 'user', 'content': 'Where is my refund?'},
            {'role': 'assistant', 'content': 'On its way.'},
        ),
        project='support-agent',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='production',
        trace_type='llm',
    )


def test_conversation_without_spans_url_has_no_spans_tab():
    html = trace_conversation(_record(), spans_url=None)

    assert 'Where is my refund?' in html
    assert 'fd-spans' not in html
    assert 'Conversation</button>' in html


def test_conversation_spans_tab_loads_the_given_url():
    html = trace_conversation(_record(), spans_url='/insights/r/trace-spans?trace_id=trace-1')

    assert 'hx-get="/insights/r/trace-spans?trace_id=trace-1"' in html
    assert '/find/trace-spans' not in html


def test_unavailable_conversation_states_its_reason():
    html = unavailable_conversation(reason='The snapshot file is gone.', spans_url=None)

    assert 'Conversation unavailable' in html
    assert 'The snapshot file is gone.' in html
    assert 'fd-spans' not in html


def test_explorer_drawers_still_point_spans_at_find_route():
    detail = TraceDetail(trace=_record(), projection=None, classification=None)

    assert 'hx-get="/find/trace-spans?trace_id=trace-1"' in drawer(detail, traces_layout=True)
    assert 'hx-get="/find/trace-spans?trace_id=trace-1"' in missing_trace_drawer('trace-1', traces_layout=True)


@pytest.mark.asyncio
async def test_span_fragment_reports_a_span_lookup_failure():
    async def load_spans(trace_id: str) -> list[object]:
        raise RuntimeError('boom')

    async def load_error(trace_id: str, spans: object) -> str | None:
        return None

    html = await span_tree_fragment('trace-1', load_spans=load_spans, load_first_error_message=load_error)

    assert 'Could not load spans. Try again.' in html


@pytest.mark.asyncio
async def test_span_fragment_renders_the_tree_when_the_error_lookup_fails():
    span = SimpleNamespace(
        span_id='s1', parent_span_id=None, type='llm', name='chat-completion', duration_ms=12, usage=None, status='ok'
    )

    async def load_spans(trace_id: str) -> list[object]:
        return [span]

    async def load_error(trace_id: str, spans: object) -> str | None:
        raise RuntimeError('status lookup down')

    html = await span_tree_fragment('trace-1', load_spans=load_spans, load_first_error_message=load_error)

    assert 'chat-completion' in html
