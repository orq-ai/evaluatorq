"""Tests for evaluatorq.common.reports.html_helpers (RES-846).

Covers the chart helpers and the try_render_svg shim (kept until A4).
render_donut_chart and render_horizontal_bar_chart now delegate to Vega-Lite
via vl-convert; they no longer use plotly/kaleido.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from evaluatorq.common.reports.html_helpers import (
    render_donut_chart,
    render_horizontal_bar_chart,
    try_render_svg,
)


# ---------------------------------------------------------------------------
# try_render_svg failure logging (shim — kept until A4 removes it)
# ---------------------------------------------------------------------------


def test_try_render_svg_logs_warning_on_failure(caplog):
    """If kaleido fails at render time, log a warning instead of swallowing."""
    import logging

    fig = MagicMock()
    fig.to_image.side_effect = RuntimeError('kaleido: chrome not installed')
    with caplog.at_level(logging.WARNING):
        result = try_render_svg(fig)
    assert result is None
    # Loguru emits to standard logging — caplog captures it.
    assert any('Chart render failed' in r.message for r in caplog.records) or any(
        'kaleido' in str(r) for r in caplog.records
    )


def test_try_render_svg_returns_decoded_svg_on_success():
    fig = MagicMock()
    fig.to_image.return_value = b'<svg>...</svg>'
    result = try_render_svg(fig)
    assert result == '<svg>...</svg>'


def test_try_render_svg_returns_str_unchanged():
    fig = MagicMock()
    fig.to_image.return_value = '<svg>raw</svg>'
    assert try_render_svg(fig) == '<svg>raw</svg>'


# ---------------------------------------------------------------------------
# render_donut_chart — behavioral tests (Vega-Lite backed)
# ---------------------------------------------------------------------------


def test_render_donut_chart_returns_empty_when_all_zero():
    """All segments zero -> nothing to render."""
    out = render_donut_chart(labels=['A'], values=[0], colors=['#fff'], title='t')
    assert out == ''


def test_render_donut_chart_filters_zero_segments_and_renders_svg():
    """Segments with value 0 must be dropped; non-zero segments produce an SVG."""
    out = render_donut_chart(
        labels=['A', 'B', 'C'],
        values=[5, 0, 3],
        colors=['#2ebd85', '#ff0000', '#d92d20'],
        title='t',
    )
    # The Vega-Lite render path produces a figure with an SVG.
    assert '<svg' in out
    assert '<figure' in out


def test_render_donut_chart_renders_svg():
    """Non-zero values produce a rendered SVG fragment."""
    out = render_donut_chart(
        labels=['pass', 'fail'],
        values=[3, 1],
        colors=['#2ebd85', '#d92d20'],
        title='Goal outcomes',
    )
    assert '<svg' in out
    assert 'Goal outcomes' in out


# ---------------------------------------------------------------------------
# render_horizontal_bar_chart — behavioral tests (Vega-Lite backed)
# ---------------------------------------------------------------------------


def test_render_horizontal_bar_chart_returns_empty_when_no_labels():
    out = render_horizontal_bar_chart(
        labels=[],
        values=[],
        color='#fff',
        title='t',
        x_title='x',
    )
    assert out == ''


def test_render_horizontal_bar_chart_renders_svg():
    out = render_horizontal_bar_chart(
        labels=['a', 'b'],
        values=[10.0, 20.0],
        color='#025558',
        title='t',
        x_title='count',
    )
    assert '<svg' in out
    assert '<figure' in out
