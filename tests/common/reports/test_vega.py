# tests/common/reports/test_vega.py
from __future__ import annotations

from evaluatorq.common.reports.vega import (
    ORQ_VL_CONFIG,
    render_embed,
    render_svg,
    vl_available,
    vl_bar_h,
    vl_donut,
    vl_grouped_bar,
    vl_heatmap,
    vl_histogram,
    vl_line,
    vl_sparkline,
)
from evaluatorq.common.reports.palette import COLORS
from evaluatorq.common.reports.html_helpers import scale_color
from evaluatorq.common.reports.palette import ORQ_SCALE_HEAT


def test_vl_available_true():
    assert vl_available() is True  # vl-convert in dev group


def test_config_uses_palette():
    assert COLORS['orange_300'] in str(ORQ_VL_CONFIG)


def test_render_svg_returns_svg():
    spec = {
        'mark': 'bar',
        'data': {'values': [{'a': 'x', 'b': 3}]},
        'encoding': {
            'x': {'field': 'a', 'type': 'nominal'},
            'y': {'field': 'b', 'type': 'quantitative'},
        },
    }
    assert '<svg' in render_svg(spec)


def test_render_svg_empty():
    assert render_svg({}) == ''


def test_render_embed_empty():
    assert render_embed({}, 'c1') == ''


def test_render_embed_emits_div_and_view_registry():
    spec = {
        'mark': 'bar',
        'data': {'values': [{'a': 'x', 'b': 1}]},
        'encoding': {
            'x': {'field': 'a', 'type': 'nominal'},
            'y': {'field': 'b', 'type': 'quantitative'},
        },
    }
    html = render_embed(spec, 'chart-42')
    assert 'id="chart-42"' in html
    assert 'vegaEmbed' in html
    assert 'window.__orqVegaViews' in html
    assert 'data-vega-for' in html


# ---------------------------------------------------------------------------
# A2: spec builder tests
# ---------------------------------------------------------------------------


def test_bar_h_value_labels_layer_and_render():
    spec = vl_bar_h(
        labels=['x', 'y'],
        values=[0.5, 0.8],
        color=COLORS['teal_400'],
        x_title='Score',
        value_labels=['50%', '80%'],
    )
    assert spec  # not empty
    # must be a layered spec (bar + text)
    assert 'layer' in spec
    # value labels present in data
    assert any(r.get('text') == '50%' for r in spec['data']['values'])
    assert '<svg' in render_svg(spec)


def test_bar_h_empty():
    assert vl_bar_h(labels=[], values=[], color=COLORS['teal_400'], x_title='X') == {}


def test_bar_h_per_bar_colors_are_literal_fills():
    # colors= gives one literal hex per bar (color.scale=None, no re-map).
    spec = vl_bar_h(
        labels=['critical', 'low'],
        values=[3, 1],
        color=COLORS['teal_400'],
        x_title='Count',
        colors=[COLORS['red_400'], COLORS['success_400']],
    )
    fills = {r['fill'] for r in spec['data']['values']}
    assert fills == {COLORS['red_400'], COLORS['success_400']}
    bar_layer = spec['layer'][0]
    assert bar_layer['encoding']['color']['scale'] is None
    out = render_svg(spec).lower()
    assert COLORS['red_400'].lower() in out and COLORS['success_400'].lower() in out


def test_donut_center_label():
    spec = vl_donut(
        labels=['a', 'b'],
        values=[0.3, 0.7],
        colors=[COLORS['success_400'], COLORS['red_400']],
        center_label='70%',
    )
    assert spec
    # center label means a layer spec with text layer
    assert 'layer' in spec
    assert any(layer.get('data', {}).get('values', [{}])[0].get('t') == '70%' for layer in spec['layer'])
    assert '<svg' in render_svg(spec)


def test_donut_empty():
    assert vl_donut(labels=[], values=[], colors=[]) == {}


def test_heatmap_literal_colors_and_safety():
    colors_grid = [[COLORS['success_400'], COLORS['red_400']]]
    texts_grid = [['0.1', '0.9']]
    safety_grid = [[False, True]]
    spec = vl_heatmap(
        x_labels=['p1', 'p2'],
        y_labels=['s1'],
        cell_colors=colors_grid,
        cell_texts=texts_grid,
        safety_mask=safety_grid,
    )
    assert spec
    # literal color must appear in the data (scale=None means precomputed)
    assert any(r.get('fill') == COLORS['success_400'] for r in spec['data']['values'])
    # safety flag set on cell
    assert any(r.get('safety') is True for r in spec['data']['values'])
    # color encoding uses scale=None
    for layer in spec.get('layer', []):
        enc = layer.get('encoding', {})
        if 'color' in enc:
            assert enc['color']['scale'] is None
            break
    assert '<svg' in render_svg(spec)


def test_heatmap_empty():
    assert vl_heatmap(x_labels=[], y_labels=['s'], cell_colors=[], cell_texts=[]) == {}


def test_histogram_mean_rule_and_prebin():
    spec = vl_histogram(values=[0.1, 0.2, 0.8, 0.9], bins=4, mean=0.5)
    assert spec
    # layers: bar + text + mean rule = 3
    assert len(spec['layer']) == 3
    # mean rule layer uses 'rule' mark
    mean_layer = spec['layer'][2]
    assert mean_layer['mark']['type'] == 'rule'
    # data in base spec contains pre-binned rows
    assert 'c' in spec['data']['values'][0]
    assert '<svg' in render_svg(spec)


def test_histogram_no_mean():
    spec = vl_histogram(values=[0.5, 0.6], bins=2)
    assert len(spec['layer']) == 2  # bar + text only


def test_histogram_empty():
    assert vl_histogram(values=[], bins=5) == {}


def test_line_keeps_nulls_and_breaks():
    spec = vl_line(x_labels=['t1', 't2', 't3'], series=[('q', [0.5, None, 0.8])])
    # null preserved in values
    assert any(r.get('v') is None for r in spec['data']['values'])
    # mark uses invalid config for gap breaks
    assert 'invalid' in spec['mark']
    assert '<svg' in render_svg(spec)


def test_line_colourblind_encoding():
    spec = vl_line(x_labels=['t1', 't2'], series=[('a', [0.2, 0.4]), ('b', [0.6, 0.8])])
    assert 'strokeDash' in spec['encoding']
    assert 'shape' in spec['encoding']


def test_line_empty():
    assert vl_line(x_labels=[], series=[]) == {}


def test_grouped_bar_renders_multi_series():
    spec = vl_grouped_bar(
        categories=['c1', 'c2'],
        series=[('agent-a', [0.2, 0.5]), ('agent-b', [0.4, 0.1])],
        x_title='ASR',
    )
    assert spec['encoding'].get('yOffset', {}).get('field') == 'series'
    assert '<svg' in render_svg(spec)


def test_grouped_bar_empty():
    assert vl_grouped_bar(categories=[], series=[], x_title='X') == {}


def test_sparkline_basic():
    spec = vl_sparkline(values=[1.0, 2.0, 3.0])
    assert spec
    assert spec['mark']['type'] == 'bar'
    assert len(spec['data']['values']) == 3
    assert '<svg' in render_svg(spec)


def test_sparkline_empty():
    assert vl_sparkline(values=[]) == {}
