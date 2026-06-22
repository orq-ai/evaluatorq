# tests/common/reports/test_vega.py
from __future__ import annotations

from evaluatorq.common.reports.vega import ORQ_VL_CONFIG, render_embed, render_svg, vl_available
from evaluatorq.common.reports.palette import COLORS


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
