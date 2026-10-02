from __future__ import annotations

from evaluatorq.dashboard.trace_finder.search_views import search_page_html, search_fragment
from evaluatorq.trace_finder.models import RunSnapshot
from evaluatorq.trace_finder.settings import DashboardSettings


def test_search_page_has_question_first_controls_and_marks_surface() -> None:
    html = search_page_html(
        RunSnapshot(),
        DashboardSettings(window_days=5, limit=200, parallelism=12),
        api_available=True,
    )

    assert 'hx-vals=\'{"surface":"search"}\'' in html
    assert 'hx-post="/find/run"' in html
    assert 'name="window_days"' in html
    assert 'name="limit"' in html
    assert 'name="parallelism"' in html
    assert 'Review first' in html and 'Immediate' in html
    assert 'Within results' not in html
    assert 'Load traces' not in html


def test_running_search_fragment_polls_search_surface() -> None:
    fragment = search_fragment(
        RunSnapshot(state='classifying'),
        DashboardSettings(window_days=7, limit=500, parallelism=100),
    )

    assert 'hx-get="/find/poll?surface=search"' in fragment
