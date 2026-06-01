"""Tests for evaluatorq.common.reports.html_helpers (RES-846).

Covers the chart helpers without requiring plotly/kaleido at test time —
real chart code never runs in CI because plotly is optional. We mock
plotly's Figure and fig.to_image to drive the filtering/guard branches
in render_donut_chart and render_horizontal_bar_chart.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from evaluatorq.common.reports.html_helpers import (
    charts_available,
    render_donut_chart,
    render_horizontal_bar_chart,
    try_render_svg,
)


# ---------------------------------------------------------------------------
# try_render_svg failure logging
# ---------------------------------------------------------------------------


def test_try_render_svg_logs_warning_on_failure(caplog):
    """If kaleido fails at render time, log a warning instead of swallowing."""
    import logging

    fig = MagicMock()
    fig.to_image.side_effect = RuntimeError("kaleido: chrome not installed")
    with caplog.at_level(logging.WARNING):
        result = try_render_svg(fig)
    assert result is None
    # Loguru emits to standard logging — caplog captures it.
    assert any("Chart render failed" in r.message for r in caplog.records) or \
           any("kaleido" in str(r) for r in caplog.records)


def test_try_render_svg_returns_decoded_svg_on_success():
    fig = MagicMock()
    fig.to_image.return_value = b"<svg>...</svg>"
    result = try_render_svg(fig)
    assert result == "<svg>...</svg>"


def test_try_render_svg_returns_str_unchanged():
    fig = MagicMock()
    fig.to_image.return_value = "<svg>raw</svg>"
    assert try_render_svg(fig) == "<svg>raw</svg>"


# ---------------------------------------------------------------------------
# render_donut_chart filtering / guards
# ---------------------------------------------------------------------------


def test_render_donut_chart_returns_empty_when_charts_unavailable():
    """When plotly is not installed the helper returns "" silently."""
    with patch("evaluatorq.common.reports.html_helpers.charts_available", return_value=False):
        out = render_donut_chart(
            labels=["A"], values=[1], colors=["#fff"], title="t",
        )
    assert out == ""


def test_render_donut_chart_filters_zero_segments():
    """Segments with value 0 must be dropped before rendering."""
    if not charts_available():
        # No plotly installed locally — exercise via mock.
        with patch(
            "evaluatorq.common.reports.html_helpers.charts_available",
            return_value=True,
        ), patch(
            "evaluatorq.common.reports.html_helpers.try_render_svg",
            return_value="<svg/>",
        ), patch.dict("sys.modules"):
            # Provide a fake plotly module so the import inside the function works.
            import sys
            fake_go = MagicMock()
            fake_plotly = MagicMock()
            fake_plotly.graph_objects = fake_go
            sys.modules["plotly"] = fake_plotly
            sys.modules["plotly.graph_objects"] = fake_go

            out = render_donut_chart(
                labels=["A", "B", "C"],
                values=[5, 0, 3],
                colors=["#fff", "#000", "#aaa"],
                title="t",
            )
            assert "<svg/>" in out
            # Verify the Pie was called with only the non-zero entries.
            pie_call = fake_go.Pie.call_args
            assert pie_call.kwargs["labels"] == ["A", "C"]
            assert pie_call.kwargs["values"] == [5, 3]


def test_render_donut_chart_returns_empty_when_all_zero():
    """All segments zero -> nothing to render."""
    with patch(
        "evaluatorq.common.reports.html_helpers.charts_available",
        return_value=True,
    ):
        out = render_donut_chart(
            labels=["A"], values=[0], colors=["#fff"], title="t",
        )
    assert out == ""


# ---------------------------------------------------------------------------
# render_horizontal_bar_chart guards
# ---------------------------------------------------------------------------


def test_render_horizontal_bar_chart_returns_empty_when_no_labels():
    with patch(
        "evaluatorq.common.reports.html_helpers.charts_available",
        return_value=True,
    ):
        out = render_horizontal_bar_chart(
            labels=[], values=[], color="#fff", title="t", x_title="x",
        )
    assert out == ""


def test_render_horizontal_bar_chart_returns_empty_when_charts_unavailable():
    with patch(
        "evaluatorq.common.reports.html_helpers.charts_available",
        return_value=False,
    ):
        out = render_horizontal_bar_chart(
            labels=["a"], values=[1.0], color="#fff", title="t", x_title="x",
        )
    assert out == ""
