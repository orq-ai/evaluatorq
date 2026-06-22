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


def vl_available() -> bool:
    """Return True if vl-convert-python is importable at runtime."""
    try:
        import vl_convert  # noqa: F401

        return True
    except ImportError:
        return False


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
        logger.warning('vl-convert unavailable; chart omitted from report.')
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
    - A ``<script>`` that registers the spec on ``window.__orqVegaViews`` so a
      single page-level loader can drive all charts with one ``vegaEmbed`` call.

    Returns ``''`` when *spec* is empty.
    """
    if not spec:
        return ''
    spec_json = json.dumps(_finalize(spec))
    safe = _html.escape(dom_id, quote=True)
    return (
        f'<div id="{safe}" class="vega-chart"></div>\n'
        f'<script>\n'
        f'(window.__orqVegaViews = window.__orqVegaViews || []).push('
        f'{{id: "{safe}", spec: {spec_json}}}'
        f');\n'
        f'</script>'
    )
