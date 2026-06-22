"""Vega-Lite renderer module.

Provides:
- ``ORQ_VL_CONFIG``: brand-aligned Vega-Lite config dict (injected into every spec).
- ``vl_available()``: runtime guard — returns True when vl-convert-python is importable.
- ``render_svg(spec)``: render a Vega-Lite spec to an SVG string via vl-convert.
- ``render_embed(spec, dom_id)``: return an HTML fragment that self-registers via
  ``window.__orqVegaViews`` for deferred client-side Vega-Embed rendering.

All functions degrade gracefully (return ``''``) when the spec is empty or when
vl-convert-python is not installed.
"""

from __future__ import annotations

import functools
import html as _html
import json
from typing import Any

from loguru import logger

from evaluatorq.common.reports.palette import COLORS, QUALITATIVE

# ---------------------------------------------------------------------------
# Brand-aligned Vega-Lite config (injected into every finalized spec)
# ---------------------------------------------------------------------------

ORQ_VL_CONFIG: dict[str, Any] = {
    'background': 'transparent',
    'font': 'Inter, system-ui, -apple-system, sans-serif',
    'axis': {
        'labelColor': COLORS['ink_700'],
        'titleColor': COLORS['ink_700'],
        'gridColor': COLORS['sand_400'],
        'domainColor': COLORS['sand_400'],
        'tickColor': COLORS['sand_400'],
    },
    'legend': {'labelColor': COLORS['ink_700'], 'titleColor': COLORS['ink_700']},
    'view': {'stroke': 'transparent'},
    'range': {'category': QUALITATIVE},
}


@functools.lru_cache(maxsize=1)
def vl_available() -> bool:
    """Return True if vl-convert-python is importable at runtime (cached)."""
    try:
        import vl_convert  # noqa: F401

        return True
    except ImportError:
        return False


# Warn at most once per process when charts are silently omitted for a missing dep.
_VL_UNAVAILABLE_WARNED = False


def _finalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Merge ORQ theme config into a Vega-Lite spec and ensure ``$schema`` is set."""
    merged = dict(spec)
    merged['config'] = {**ORQ_VL_CONFIG, **spec.get('config', {})}
    merged.setdefault('$schema', 'https://vega.github.io/schema/vega-lite/v5.json')
    return merged


def render_svg(spec: dict[str, Any]) -> str:
    """Render a Vega-Lite spec to an SVG string.

    Returns ``''`` when *spec* is empty or when vl-convert-python is unavailable.
    On render failure the exception is logged at WARNING level and ``''`` is returned.
    """
    if not spec:
        return ''
    if not vl_available():
        global _VL_UNAVAILABLE_WARNED
        if not _VL_UNAVAILABLE_WARNED:
            logger.warning('vl-convert-python not installed; charts omitted from reports.')
            _VL_UNAVAILABLE_WARNED = True
        return ''
    try:
        import vl_convert as vlc

        return vlc.vegalite_to_svg(json.dumps(_finalize(spec)))
    except Exception:
        logger.opt(exception=True).warning('Vega-Lite SVG render failed; chart omitted.')
        return ''


def render_embed(spec: dict[str, Any], dom_id: str) -> str:
    """Return an HTML fragment for client-side Vega-Embed rendering.

    The fragment contains:
    - A ``<div>`` with *dom_id* as its ``id``.
    - A ``<script>`` that immediately calls ``vegaEmbed`` if the library is on the
      page, and registers the resulting view on ``window.__orqVegaViews[dom_id]``
      for external access (e.g. teardown).

    Returns ``''`` when *spec* is empty.
    """
    if not spec:
        return ''
    # Escape ``</`` so a data value containing ``</script>`` cannot break out of
    # the inline JSON island (json.dumps does not do this).
    spec_json = json.dumps(_finalize(spec)).replace('</', '<\\/')
    safe = _html.escape(dom_id, quote=True)
    return (
        f'<div id="{safe}" class="vega-chart"></div>'
        f'<script type="application/json" data-vega-for="{safe}">{spec_json}</script>'
        '<script>(function(){window.__orqVegaViews=window.__orqVegaViews||{};'
        f'var el=document.getElementById("{safe}");'
        f'var s=JSON.parse(document.querySelector(\'[data-vega-for="{safe}"]\').textContent);'
        f'if(window.vegaEmbed&&el){{vegaEmbed(el,s,{{actions:false}}).then(function(r){{'
        f'window.__orqVegaViews["{safe}"]=r;}});}}}})();</script>'
    )


# ---------------------------------------------------------------------------
# Spec builders — return a Vega-Lite v5 dict, or {} when there is nothing to draw.
# Specs carry NO title (the figcaption wrapper owns it).
# ---------------------------------------------------------------------------


def vl_bar_h(
    *,
    labels: list[str],
    values: list[float],
    color: str,
    x_title: str,
    value_labels: list[str] | None = None,
    colors: list[str] | None = None,
) -> dict[str, Any]:
    """Horizontal bar chart with an optional text-label overlay layer.

    ``colors`` gives one literal hex per bar (``color.scale=None``, no re-map) to
    carry a per-bar signal such as severity; ``color`` is the uniform fallback.
    """
    if not labels:
        return {}
    rows = [
        {
            'label': label,
            'value': v,
            'text': (value_labels[i] if i < len(value_labels) else f'{v:g}') if value_labels else f'{v:g}',
            'fill': (colors[i] if i < len(colors) else color) if colors else color,
        }
        for i, (label, v) in enumerate(zip(labels, values, strict=False))
    ]
    base: dict[str, Any] = {
        'data': {'values': rows},
        'encoding': {
            'y': {'field': 'label', 'type': 'nominal', 'sort': None, 'title': None},
            'x': {'field': 'value', 'type': 'quantitative', 'title': x_title},
        },
        'width': 420,
        'height': {'step': 24},
    }
    bar_layer: dict[str, Any] = {
        'mark': {'type': 'bar'},
        'encoding': {
            'y': {'field': 'label', 'type': 'nominal', 'sort': None, 'title': None},
            'x': {'field': 'value', 'type': 'quantitative', 'title': x_title},
            'color': {'field': 'fill', 'type': 'nominal', 'scale': None, 'legend': None},
        },
    }
    text_layer: dict[str, Any] = {
        'mark': {'type': 'text', 'align': 'left', 'dx': 4, 'color': COLORS['ink_700']},
        'encoding': {
            'y': {'field': 'label', 'type': 'nominal', 'sort': None},
            'x': {'field': 'value', 'type': 'quantitative'},
            'text': {'field': 'text', 'type': 'nominal'},
        },
    }
    return {**base, 'layer': [bar_layer, text_layer]}


def vl_donut(
    *,
    labels: list[str],
    values: list[float],
    colors: list[str],
    center_label: str = '',
) -> dict[str, Any]:
    """Donut arc chart with an optional center text label."""
    if not labels:
        return {}
    rows = [{'label': label, 'value': v, 'color': c} for label, v, c in zip(labels, values, colors, strict=False)]
    arc: dict[str, Any] = {
        'data': {'values': rows},
        'mark': {'type': 'arc', 'innerRadius': 60, 'tooltip': True},
        'encoding': {
            'theta': {'field': 'value', 'type': 'quantitative'},
            'color': {'field': 'color', 'type': 'nominal', 'scale': None, 'legend': None},
            'order': {'field': 'value', 'type': 'quantitative', 'sort': 'descending'},
        },
    }
    if not center_label:
        return {**arc, 'width': 240, 'height': 240}
    return {
        'width': 240,
        'height': 240,
        'layer': [
            arc,
            {
                'data': {'values': [{'t': center_label}]},
                'mark': {'type': 'text', 'fontSize': 20, 'fontWeight': 'bold', 'color': COLORS['ink_700']},
                'encoding': {'text': {'field': 't', 'type': 'nominal'}},
            },
        ],
    }


def vl_heatmap(
    *,
    x_labels: list[str],
    y_labels: list[str],
    cell_colors: list[list[str]],
    cell_texts: list[list[str]],
    safety_mask: list[list[bool]] | None = None,
) -> dict[str, Any]:
    """Heatmap with literal precomputed cell colors (color.scale=None) and optional safety highlight."""
    if not x_labels or not y_labels:
        return {}
    rows: list[dict[str, Any]] = []
    for yi, y in enumerate(y_labels):
        for xi, x in enumerate(x_labels):
            rows.append({
                'x': x,
                'y': y,
                'fill': cell_colors[yi][xi],
                'text': cell_texts[yi][xi],
                'safety': bool(safety_mask and safety_mask[yi][xi]),
            })
    return {
        'data': {'values': rows},
        'layer': [
            {
                'mark': {'type': 'rect', 'tooltip': True},
                'encoding': {
                    'x': {'field': 'x', 'type': 'nominal', 'title': None},
                    'y': {'field': 'y', 'type': 'nominal', 'title': None},
                    'color': {'field': 'fill', 'type': 'nominal', 'scale': None, 'legend': None},
                    'stroke': {
                        'condition': {'test': 'datum.safety', 'value': COLORS['red_400']},
                        'value': None,
                    },
                    'strokeWidth': {
                        'condition': {'test': 'datum.safety', 'value': 3},
                        'value': 0,
                    },
                },
            },
            {
                'mark': {'type': 'text', 'fontSize': 11},
                'encoding': {
                    'x': {'field': 'x', 'type': 'nominal'},
                    'y': {'field': 'y', 'type': 'nominal'},
                    'text': {'field': 'text', 'type': 'nominal'},
                },
            },
        ],
        'width': 420,
        'height': 200,
    }


def vl_histogram(
    *,
    values: list[float],
    bins: int,
    mean: float | None = None,
) -> dict[str, Any]:
    """Histogram with counts pre-computed in Python (deterministic; no dual-aggregate).

    Includes a bar layer, count text labels, and an optional mean rule.
    """
    if not values or bins <= 0:
        return {}
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int(float(v) * bins)))
        counts[idx] += 1
    bin_rows = [
        {
            'lo': i / bins,
            'hi': (i + 1) / bins,
            'mid': (i + 0.5) / bins,
            'c': counts[i],
        }
        for i in range(bins)
    ]
    base: dict[str, Any] = {'data': {'values': bin_rows}}
    bar: dict[str, Any] = {
        'mark': {'type': 'bar', 'color': COLORS['teal_400'], 'tooltip': True},
        'encoding': {
            'x': {'field': 'lo', 'type': 'quantitative', 'title': None},
            'x2': {'field': 'hi'},
            'y': {'field': 'c', 'type': 'quantitative', 'title': 'Count'},
        },
    }
    text: dict[str, Any] = {
        'mark': {'type': 'text', 'dy': -4, 'color': COLORS['ink_700']},
        'encoding': {
            'x': {'field': 'mid', 'type': 'quantitative'},
            'y': {'field': 'c', 'type': 'quantitative'},
            'text': {'field': 'c', 'type': 'quantitative'},
        },
    }
    layers: list[dict[str, Any]] = [bar, text]
    if mean is not None:
        layers.append({
            'data': {'values': [{'m': float(mean)}]},
            'mark': {'type': 'rule', 'strokeDash': [4, 4], 'color': COLORS['orange_300']},
            'encoding': {'x': {'field': 'm', 'type': 'quantitative'}},
        })
    return {**base, 'layer': layers, 'width': 420, 'height': 200}


def vl_line(
    *,
    x_labels: list[str],
    series: list[tuple[str, list[float | None]]],
) -> dict[str, Any]:
    """Multi-series line chart.

    Nulls are kept in the data and ``mark.invalid`` is set so Vega-Lite breaks
    the line across gaps rather than connecting across them.  Each series gets
    a distinct ``strokeDash`` and point ``shape`` for colour-blind accessibility.
    """
    if not x_labels or not series:
        return {}
    # Flatten to long-form rows, preserving nulls for gap detection.
    rows: list[dict[str, Any]] = []
    for name, ys in series:
        for i, x in enumerate(x_labels):
            rows.append({'x': x, 'series': name, 'v': ys[i] if i < len(ys) else None})
    return {
        'data': {'values': rows},
        'mark': {
            'type': 'line',
            'point': {'filled': False},
            # 'break-paths-keep-domains' is VL v5.18+ — vl-convert ships an earlier
            # build that only supports 'filter'.  'filter' drops null points so the
            # line still breaks at gaps; upgrade when vl-convert catches up.
            'invalid': 'filter',
        },
        'encoding': {
            'x': {'field': 'x', 'type': 'nominal', 'title': None},
            'y': {'field': 'v', 'type': 'quantitative', 'title': None},
            'color': {'field': 'series', 'type': 'nominal', 'legend': {'title': None}},
            'strokeDash': {'field': 'series', 'type': 'nominal'},
            'shape': {'field': 'series', 'type': 'nominal'},
        },
        'width': 420,
        'height': 200,
    }


def vl_grouped_bar(
    *,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    x_title: str,
) -> dict[str, Any]:
    """Grouped horizontal bar chart for multi-series agent-comparison.

    Each series produces one bar per category, offset via ``yOffset``.
    ``series`` is a list of ``(name, values_per_category)`` tuples.
    """
    if not categories or not series:
        return {}
    rows = [
        {'cat': categories[i], 'series': name, 'value': vals[i]}
        for name, vals in series
        for i in range(len(categories))
        if i < len(vals)
    ]
    return {
        'data': {'values': rows},
        'mark': {'type': 'bar', 'tooltip': True},
        'encoding': {
            'y': {'field': 'cat', 'type': 'nominal', 'sort': None, 'title': None},
            'x': {'field': 'value', 'type': 'quantitative', 'title': x_title},
            'yOffset': {'field': 'series', 'type': 'nominal'},
            'color': {'field': 'series', 'type': 'nominal', 'legend': {'title': None}},
        },
        'width': 420,
        'height': {'step': 16},
    }


def vl_sparkline(*, values: list[float]) -> dict[str, Any]:
    """Bare mini bar chart for inline use in table rows."""
    if not values:
        return {}
    return {
        'data': {'values': [{'i': i, 'v': float(v)} for i, v in enumerate(values)]},
        'mark': {'type': 'bar', 'color': COLORS['teal_400']},
        'encoding': {
            'x': {'field': 'i', 'type': 'ordinal', 'axis': None},
            'y': {'field': 'v', 'type': 'quantitative', 'axis': None},
        },
        'width': 80,
        'height': 20,
    }
