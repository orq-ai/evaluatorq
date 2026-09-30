from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from evaluatorq.dashboard.trace_finder.views import drawer
from evaluatorq.dashboard.trace_finder.views import span_tree
from evaluatorq.trace_finder import TraceDetail
from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.rows import TraceRow


def test_trace_drawer_shows_formatted_and_unknown_trace_duration() -> None:
    trace = TraceRecord(
        schema_version=1,
        trace_id='t',
        span_id='s',
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
        project='p',
        model='gpt-5.6-luna',
        provider='openai',
        status='error',
        product='chat',
        trace_type='agent',
        messages=({'role': 'user', 'content': 'hello'},),
    )
    detail = TraceDetail(trace=trace, projection=None, classification=None)

    formatted = drawer(detail, row=TraceRow(trace_id='t', duration_ms=1234), traces_layout=True)
    unknown = drawer(detail, row=TraceRow(trace_id='t'), traces_layout=True)

    assert 'Duration 1.2s' in formatted
    assert 'Duration —' in unknown


def test_span_tree_and_lazy_error_lookup_share_loose_status_shapes() -> None:
    tree = span_tree(
        't',
        [
            {'status': 'failed'},
            {'span_id': 'status-code', 'status_code': 'ERROR'},
            SimpleNamespace(span_id='object-status', status=SimpleNamespace(code='failed')),
        ],
        first_error_message='detail loaded for status-code',
    )

    assert 'fd-span-error' in tree
    assert 'fd-span-first-error' in tree
    assert 'detail loaded for status-code' in tree
    assert tree.index('fd-span-first-error') < tree.index('status-code')
    assert tree.count('fd-span-first-error') == 1
