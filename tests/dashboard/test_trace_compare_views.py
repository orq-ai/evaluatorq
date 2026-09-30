from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.dashboard.trace_finder.compare_views import comparison_panel
from evaluatorq.trace_finder.rows import TraceRow


def test_comparison_panel_shows_requested_metrics_and_drawer_links() -> None:
    first = TraceRow(
        trace_id='trace/one',
        name='First trace',
        status='ok',
        started_at=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        duration_ms=1250,
        models=('gpt-5.6-luna',),
        tokens_in=1200,
        tokens_out=80,
        cached_tokens=300,
        cost_total=0.25,
    )
    second = TraceRow(trace_id='trace-two', status='failed', duration_ms=45, tokens_in=10, cached_tokens=2)

    html = comparison_panel(first, second)

    for label in ('Status / error', 'Started', 'Duration', 'Model', 'Input tokens', 'Output tokens', 'Cache reads', 'Cost'):
        assert label in html
    assert 'Success' in html
    assert 'Error (failed)' in html
    assert 'gpt-5.6-luna' in html
    assert '1.2k' in html
    assert '80' in html
    assert '300' in html
    assert '$0.25' in html
    assert 'href="/find/trace/trace%2Fone?surface=traces"' in html
    assert 'Open First trace drawer' in html
    assert 'Open trace-two drawer' in html


def test_comparison_panel_escapes_untrusted_text_and_marks_unknown_values() -> None:
    first = TraceRow(
        trace_id='id"><script>',
        name='Name <script>&',
        status='pending<&',
        models=('model <tag>&',),
        currency='X<&',
        cost_total=1.0,
    )
    second = TraceRow(trace_id='second', status='failed')

    html = comparison_panel(first, second)

    assert '<script>' not in html
    assert 'Name &lt;script&gt;&amp;' in html
    assert 'pending&lt;&amp;' in html
    assert 'Error (failed)' in html
    assert 'model &lt;tag&gt;&amp;' in html
    assert '1.0000 X&lt;&amp;' in html
    assert html.count('—') >= 7
    assert 'id%22%3E%3Cscript%3E' in html
