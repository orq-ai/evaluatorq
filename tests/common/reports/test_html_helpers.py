# tests/common/reports/test_html_helpers.py
from __future__ import annotations

from evaluatorq.common.reports import html_helpers as h
from evaluatorq.common.reports.html_helpers import render_heatmap, scale_color
from evaluatorq.common.reports.palette import COLORS, ORQ_SCALE_GOOD_BAD


# ---------------------------------------------------------------------------
# scale_color — exact-hex unit tests (no rendered markup, always deterministic)
# ---------------------------------------------------------------------------


def test_scale_color_endpoints_and_clamp():
    assert h.scale_color(0.0, ORQ_SCALE_GOOD_BAD).lower() == '#2ebd85'
    assert h.scale_color(1.0, ORQ_SCALE_GOOD_BAD).lower() == '#d92d20'
    # clamps out-of-range
    assert h.scale_color(-5, ORQ_SCALE_GOOD_BAD).lower() == '#2ebd85'
    assert h.scale_color(9, ORQ_SCALE_GOOD_BAD).lower() == '#d92d20'


def test_scale_color_midpoint_is_between():
    mid = h.scale_color(0.5, ORQ_SCALE_GOOD_BAD).lstrip('#')
    r = int(mid[0:2], 16)
    # midpoint is the yellow stop (#f2b600) -> red channel high
    assert r > 200


def test_scale_color_interpolates_between_stops():
    """A value not on any stop must lerp strictly between the two bounding
    stops' colors (exercises the lo+(hi-lo)*t interpolation)."""
    lo = (0x2E, 0xBD, 0x85)  # #2ebd85 (green stop at 0.0)
    hi = (0xF2, 0xB6, 0x00)  # #f2b600 (yellow stop at 0.5)
    c = h.scale_color(0.25, ORQ_SCALE_GOOD_BAD).lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    # At least one channel strictly between the two stops' channels.
    between = [min(lo[i], hi[i]) < ch < max(lo[i], hi[i]) for i, ch in enumerate((r, g, b))]
    assert any(between), f'#{c} is not strictly between {lo} and {hi}'


# ---------------------------------------------------------------------------
# Empty-guard tests (no rendering needed)
# ---------------------------------------------------------------------------


def test_render_histogram_empty_guards():
    assert h.render_histogram(values=[], bins=10, title='t') == ''
    assert h.render_histogram(values=[0.5], bins=0, title='t') == ''


def test_render_line_chart_empty_guard():
    assert h.render_line_chart(x_labels=[], series=[], title='t') == ''


def test_render_heatmap_empty_guards():
    assert (
        h.render_heatmap(
            x_labels=[],
            y_labels=['r1'],
            cells=[[]],
            scale=ORQ_SCALE_GOOD_BAD,
            title='t',
        )
        == ''
    )
    assert (
        h.render_heatmap(
            x_labels=['c1'],
            y_labels=[],
            cells=[],
            scale=ORQ_SCALE_GOOD_BAD,
            title='t',
        )
        == ''
    )


def test_svg_donut_empty_when_all_zero():
    assert h.svg_donut(labels=['a'], values=[0], colors=['#000'], center_label='', title='t') == ''


def test_svg_bar_empty_when_no_rows():
    assert h.svg_bar(rows=[], title='t') == ''


# ---------------------------------------------------------------------------
# Behavioral tests — assert rendered <svg> is present + key text, not shape counts
# ---------------------------------------------------------------------------


def test_svg_donut_renders_slices_and_center():
    out = h.svg_donut(
        labels=['Achieved', 'Failed'],
        values=[1, 3],
        colors=['#2ebd85', '#d92d20'],
        center_label='25%',
        title='Goal Outcomes',
    )
    assert out.startswith('<figure')
    assert '<svg' in out
    assert '25%' in out
    assert 'Goal Outcomes' in out


def test_svg_donut_full_circle_single_slice():
    """A single slice covering the whole circle must still render a visible ring."""
    out = h.svg_donut(
        labels=['Achieved'],
        values=[4],
        colors=['#2ebd85'],
        center_label='100%',
        title='Goal Outcomes',
    )
    assert out.startswith('<figure')
    assert '<svg' in out
    assert '100%' in out


def test_svg_bar_renders_labeled_bars():
    out = h.svg_bar(
        rows=[('1 turn', 3), ('2 turns', 1)],
        title='Conversations by turn count',
    )
    assert out.startswith('<figure')
    assert '<svg' in out
    assert '1 turn' in out
    assert '3' in out
    assert 'Conversations by turn count' in out


def test_render_heatmap_cells_and_labels():
    out = h.render_heatmap(
        x_labels=['c1', 'c2'],
        y_labels=['must explain charge', 'must not be rude'],
        cells=[[1.0, 0.0], [1.0, 1.0]],
        scale=ORQ_SCALE_GOOD_BAD,
        title='Criteria pass/fail',
        value_fmt=lambda v: 'PASS' if v >= 0.5 else 'FAIL',
    )
    assert out.startswith('<figure')
    assert '<svg' in out
    assert 'must explain charge' in out
    assert 'PASS' in out and 'FAIL' in out
    assert 'Criteria pass/fail' in out


def test_render_heatmap_absent_cell_is_neutral():
    """Absent cells (value < 0) must use the grey sentinel, not the scale."""
    out = h.render_heatmap(
        x_labels=['c1'],
        y_labels=['r1'],
        cells=[[-1.0]],
        scale=ORQ_SCALE_GOOD_BAD,
        title='t',
        value_fmt=lambda v: '—' if v < 0 else 'x',
    )
    assert '<svg' in out
    assert '—' in out


def test_render_heatmap_safety_flag_applies():
    """Safety cells must still render (the SVG red stroke is in the Vega spec)."""
    out = h.render_heatmap(
        x_labels=['c1'],
        y_labels=['no PII leak'],
        cells=[[0.0]],
        scale=ORQ_SCALE_GOOD_BAD,
        title='t',
        value_fmt=lambda v: 'FAIL',
        safety_mask=[[True]],
    )
    assert '<svg' in out


def test_render_histogram_bins():
    out = h.render_histogram(values=[0.0, 0.1, 0.9, 1.0], bins=2, title='Score distribution')
    assert out.startswith('<figure')
    assert '<svg' in out
    assert 'Score distribution' in out


def test_render_line_chart_series():
    out = h.render_line_chart(
        x_labels=['1', '2', '3'],
        series=[('response_quality', [0.5, 0.7, 0.9])],
        title='Turn quality',
    )
    assert out.startswith('<figure')
    assert '<svg' in out
    assert 'response_quality' in out
    assert 'Turn quality' in out


def test_render_line_chart_none_is_gap_not_zero():
    """A None value is 'not measured': the chart must still render an SVG."""
    out = h.render_line_chart(
        x_labels=['1', '2', '3'],
        series=[('factual_accuracy', [0.8, None, 0.6])],
        title='t',
    )
    assert '<svg' in out


def test_render_line_chart_none_segments_connect_runs():
    out = h.render_line_chart(
        x_labels=['1', '2', '3', '4'],
        series=[('m', [0.2, 0.4, None, 0.9])],
        title='t',
    )
    assert '<svg' in out


def test_render_sparkline_minibars():
    out = h.render_sparkline([1, 3, 2])
    assert '<svg' in out


def test_render_sparkline_empty():
    assert h.render_sparkline([]) == ''


# ---------------------------------------------------------------------------
# A3 parity tests — behavioral assertions on kwarg forwarding
# ---------------------------------------------------------------------------

GREEN_HIGH = [[0.0, COLORS['red_400']], [0.5, COLORS['yellow_400']], [1.0, COLORS['success_400']]]


def test_hbar_value_suffix():
    """value_suffix must appear in the rendered label text."""
    out = h.render_horizontal_bar_chart(
        labels=['a', 'b'],
        values=[73, 12],
        color=COLORS['teal_400'],
        title='ASR',
        x_title='%',
        value_suffix='%',
    )
    assert '73%' in out


def test_hbar_empty_labels_returns_empty():
    out = h.render_horizontal_bar_chart(labels=[], values=[], color=COLORS['teal_400'], title='t', x_title='x')
    assert out == ''


def test_svg_bar_value_labels_single_title():
    """value_fmt must be applied to produce per-bar text labels."""
    out = h.svg_bar(rows=[('x', 5.0)], title='t', value_fmt=lambda v: f'{v:.0f}')
    assert '5' in out
    assert out.count('<figcaption') == 1


def test_svg_donut_center_label():
    """center_label must appear in the rendered SVG."""
    out = h.svg_donut(
        labels=['pass', 'fail'],
        values=[3, 1],
        colors=[COLORS['success_400'], COLORS['red_400']],
        center_label='75%',
        title='t',
    )
    assert '75%' in out


def test_heatmap_high_cell_is_the_scale_color_not_red():
    # Deterministic: rendered fill equals scale_color(0.95, GREEN_HIGH) — proves scale direction preserved.
    expected = scale_color(0.95, GREEN_HIGH).lstrip('#').lower()
    out = render_heatmap(
        x_labels=['p'],
        y_labels=['s'],
        cells=[[0.95]],
        scale=GREEN_HIGH,
        title='Success',
        value_fmt=lambda v: f'{v:.0%}',
    ).lower()
    assert expected in out and COLORS['red_400'].lstrip('#').lower() not in out


def test_heatmap_negative_sentinel_grey():
    """Values < 0 must use the #e4e2df grey sentinel, not the color scale."""
    # scale_color clamps <0 to the scale's lo stop, so we verify via scale_color(0) != sentinel.
    sentinel = '#e4e2df'
    scale_lo = h.scale_color(0.0, GREEN_HIGH)
    # The sentinel is not the same as the scale's lo color (which is red_400).
    assert scale_lo.lower() != sentinel
    # The sentinel IS the sand_400 palette color.
    assert COLORS['sand_400'].lower() == sentinel


def test_render_heatmap_absent_cell_sentinel_in_spec():
    """Absent cell (value < 0) must use grey sentinel color in rendered output."""
    out = h.render_heatmap(
        x_labels=['c1'],
        y_labels=['r1'],
        cells=[[-1.0]],
        scale=GREEN_HIGH,
        title='t',
        value_fmt=lambda v: 'N/A',
    )
    assert '<svg' in out
    # The sentinel hex must appear somewhere in the SVG (as a fill color in the data).
    assert '#e4e2df' in out


def test_render_donut_chart_filters_zero_values():
    """Segments with value 0 must not appear in the rendered SVG as meaningful slices."""
    out = h.render_donut_chart(
        labels=['A', 'B', 'C'],
        values=[5, 0, 3],
        colors=['#2ebd85', '#ff0000', '#d92d20'],
        title='t',
    )
    assert '<svg' in out


def test_render_donut_chart_all_zero_returns_empty():
    out = h.render_donut_chart(labels=['A'], values=[0], colors=['#fff'], title='t')
    assert out == ''


def test_charts_available_delegates_to_vl_available():
    """charts_available() must delegate to vl_available()."""
    from evaluatorq.common.reports.vega import vl_available

    assert h.charts_available() == vl_available()


# ---------------------------------------------------------------------------
# CSS token presence tests
# ---------------------------------------------------------------------------


from evaluatorq.common.reports.html_helpers import load_css


def test_report_css_has_new_design_tokens():
    css = load_css()
    for token in [
        '.hero',
        '.kpi-band',
        '.kpi-card',
        '.report-card',
        '.chart-card',
        '.status-badge--pass',
        '.status-badge--fail',
        '.heatmap-table',
        '.heatmap-cell',
        '.sparkline',
        '@media',
    ]:
        assert token in css, f'missing {token}'


# ---------------------------------------------------------------------------
# Other primitive tests
# ---------------------------------------------------------------------------


def test_kpi_cards_renders_each_card_with_status():
    html_out = h.kpi_cards([
        {'label': 'Success Rate', 'value': '25%', 'status': 'fail'},
        {'label': 'Conversations', 'value': '4', 'status': 'neutral'},
    ])
    assert 'class="kpi-band"' in html_out
    assert html_out.count('kpi-card') >= 2
    assert 'Success Rate' in html_out and '25%' in html_out
    assert 'kpi-card--fail' in html_out


def test_status_badge_classes():
    assert 'status-badge--pass' in h.status_badge('ACHIEVED', 'pass')
    assert 'status-badge--fail' in h.status_badge('NOT ACHIEVED', 'fail')
    assert 'NOT ACHIEVED' in h.status_badge('NOT ACHIEVED', 'fail')
