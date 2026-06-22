"""HTML formatting helpers shared across report renderers.

Brand colors, the parameterized CSS loader, and small HTML primitives
(``esc``, ``html_table``, ``pct``, ``truncate``) live here. Chart helpers
(``scale_color``, ``svg_donut``, ``svg_bar``, ``render_heatmap``,
``render_histogram``, ``render_line_chart``, ``render_sparkline``,
``render_donut_chart``, ``render_horizontal_bar_chart``, ``kpi_cards``,
``status_badge``) build Vega-Lite specs rendered to SVG via vl-convert.
"""

from __future__ import annotations

import html
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from evaluatorq.common.reports.palette import COLORS

# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------

# COLORS is re-exported from palette so existing callers keep working.
# STATUS_COLORS uses report-level keys (success/warning/failure) distinct from
# palette.STATUS_COLORS which uses vulnerability-level keys (vulnerable/resistant/error).
STATUS_COLORS: dict[str, str] = {
    'success': COLORS['success_400'],
    'warning': COLORS['yellow_400'],
    'failure': COLORS['red_400'],
}


# ---------------------------------------------------------------------------
# CSS loading
# ---------------------------------------------------------------------------

_CSS_CACHE: dict[Path, str] = {}
_logo_cache: str | None = None


def load_logo_svg() -> str:
    """Return the orq logo SVG as an inline string, or empty string if missing."""
    global _logo_cache
    if _logo_cache is not None:
        return _logo_cache
    logo_path = Path(__file__).parent / 'assets' / 'Orq_ai_Symbol_Dark.svg'
    try:
        _logo_cache = logo_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning('orq logo asset unavailable at {} ({}); header will render without logo', logo_path, exc)
        _logo_cache = ''
    return _logo_cache


def load_css(css_path: Path | None = None) -> str:
    """Load and interpolate the shared report.css with the brand color palette.

    Uses ``string.Template`` (``$color_name``) so bare ``%`` characters in CSS
    (e.g. ``opacity: 50%``) don't raise ``ValueError`` like ``%``-formatting
    would.

    Args:
        css_path: Path to a ``.css`` file with ``$color_name`` placeholders.
            Defaults to the bundled ``common/reports/report.css``.

    Returns:
        The CSS text with brand colors substituted in.
    """
    path = css_path or Path(__file__).with_name('report.css')
    cached = _CSS_CACHE.get(path)
    if cached is not None:
        return cached
    text = Template(path.read_text(encoding='utf-8')).safe_substitute(COLORS)
    _CSS_CACHE[path] = text
    return text


# ---------------------------------------------------------------------------
# Small HTML primitives
# ---------------------------------------------------------------------------


def esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def pct(rate: float) -> str:
    """Format a float rate as a percentage string."""
    return f'{rate:.0%}'


def truncate(text: str, max_chars: int = 800) -> str:
    """Truncate long text with a plain-text marker (no Markdown)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n[truncated — full text in report JSON]'


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an HTML table. Cell strings may contain inline HTML (e.g. badges)."""
    parts = ['<table>', '<thead><tr>']
    parts.extend(f'<th>{esc(h)}</th>' for h in headers)
    parts.append('</tr></thead><tbody>')
    for row in rows:
        parts.append('<tr>')
        # data-label carries the column name so the mobile card layout
        # (td::before { content: attr(data-label) }) stays labeled.
        parts.extend(
            f'<td data-label="{esc(headers[i])}">{cell}</td>' if i < len(headers) else f'<td>{cell}</td>'
            for i, cell in enumerate(row)
        )
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Chart helpers (Plotly + kaleido optional)
# ---------------------------------------------------------------------------


def charts_available() -> bool:
    """Check whether vl-convert-python is importable (used to gate chart rendering)."""
    from evaluatorq.common.reports.vega import vl_available

    return vl_available()


def try_render_svg(fig: Any) -> str | None:
    """Render a Plotly figure as inline SVG, or ``None`` on failure.

    Failures here are not the "deps not installed" case (``charts_available``
    has already gated the call). They are real render failures — missing
    Chrome/Chromium runtime, OOM, version mismatch — and silently dropping
    them produces "the report is broken and I don't know why" output. Log a
    warning with the exception so the failure is observable.
    """
    try:
        svg_bytes = fig.to_image(format='svg', engine='kaleido')
        return svg_bytes.decode('utf-8') if isinstance(svg_bytes, bytes) else svg_bytes
    except Exception:
        logger.opt(exception=True).warning(
            'Chart render failed; chart will be omitted from the report. '
            'Check kaleido runtime (Chrome/Chromium) and plotly version.'
        )
        return None


def render_donut_chart(
    *,
    labels: list[str],
    values: list[int],
    colors: list[str],
    title: str,
) -> str:
    """Render a donut chart wrapped in a ``<figure class="chart-card">`` block.

    Builds a Vega-Lite spec rendered to SVG via vl-convert. Returns an empty
    string when vl-convert is unavailable, all values are zero, or the render
    fails.
    """
    from evaluatorq.common.reports.vega import render_svg, vl_donut

    filtered = [(lbl, v, c) for lbl, v, c in zip(labels, values, colors, strict=False) if v > 0]
    if not filtered:
        return ''
    labels_f, values_f, colors_f = zip(*filtered, strict=False)
    spec = vl_donut(labels=list(labels_f), values=list(values_f), colors=list(colors_f))
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


def render_horizontal_bar_chart(
    *,
    labels: list[str],
    values: list[float],
    color: str,
    title: str,
    x_title: str,
    value_suffix: str = '',
) -> str:
    """Render a horizontal bar chart wrapped in a ``<figure class="chart-card">`` block.

    Builds a Vega-Lite spec rendered to SVG via vl-convert. Returns an empty
    string when vl-convert is unavailable, *labels* is empty, or the render fails.
    """
    if not labels:
        return ''
    from evaluatorq.common.reports.vega import render_svg, vl_bar_h

    value_labels = [f'{v:.0f}{value_suffix}' for v in values]
    spec = vl_bar_h(labels=labels, values=values, color=color, x_title=x_title, value_labels=value_labels)
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


# ---------------------------------------------------------------------------
# Chart primitives — build Vega-Lite specs rendered to SVG via vl-convert
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*(max(0, min(255, round(c))) for c in rgb))


def scale_color(value: float, scale: list[list[float | str]]) -> str:
    """Interpolate a hex color for ``value`` in [0, 1] along a Plotly-style scale.

    ``scale`` is a list of ``[position, hex]`` stops sorted by position.
    Values outside [0, 1] are clamped to the scale endpoints.
    """
    v = max(0.0, min(1.0, float(value)))
    stops = [(float(pos), str(color)) for pos, color in scale]
    for i in range(len(stops) - 1):
        lo_pos, lo_color = stops[i]
        hi_pos, hi_color = stops[i + 1]
        if lo_pos <= v <= hi_pos:
            span = hi_pos - lo_pos or 1.0
            t = (v - lo_pos) / span
            lo = _hex_to_rgb(lo_color)
            hi = _hex_to_rgb(hi_color)
            return _rgb_to_hex((
                lo[0] + (hi[0] - lo[0]) * t,
                lo[1] + (hi[1] - lo[1]) * t,
                lo[2] + (hi[2] - lo[2]) * t,
            ))
    return stops[-1][1]


def svg_donut(
    *,
    labels: list[str],
    values: list[float],
    colors: list[str],
    center_label: str,
    title: str,
    size: int = 220,
) -> str:
    """Donut arc chart. Builds a Vega-Lite spec rendered to SVG via vl-convert.

    Returns '' when all values are zero or vl-convert is unavailable.
    The ``size`` parameter is accepted for API compatibility but Vega-Lite
    controls the rendered dimensions.
    """
    from evaluatorq.common.reports.vega import render_svg, vl_donut

    total = sum(v for v in values if v > 0)
    if total <= 0:
        return ''
    spec = vl_donut(labels=labels, values=values, colors=colors, center_label=center_label)
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


def svg_bar(
    *,
    rows: list[tuple[str, float]],
    title: str,
    color: str | None = None,
    width: int = 460,
    bar_height: int = 24,
    gap: int = 12,
    label_w: int = 150,
    value_fmt: Callable[[float], str] | None = None,
) -> str:
    """Horizontal bar chart. Builds a Vega-Lite spec rendered to SVG via vl-convert.

    Returns '' when no rows or vl-convert is unavailable. The ``value_fmt``
    callable is applied to produce per-bar text labels. The ``width``,
    ``bar_height``, ``gap``, and ``label_w`` kwargs are accepted for API
    compatibility.
    # ponytail: authored width/label_w not mapped to VL step sizing — revisit long labels clip.
    """
    if not rows:
        return ''
    from evaluatorq.common.reports.palette import COLORS as _COLORS
    from evaluatorq.common.reports.vega import render_svg, vl_bar_h

    bar_color = color or _COLORS['teal_400']
    fmt = value_fmt or (lambda v: f'{v:g}')
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    value_labels = [fmt(v) for v in values]
    spec = vl_bar_h(labels=labels, values=values, color=bar_color, x_title='', value_labels=value_labels)
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


def render_heatmap(
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    cells: Sequence[Sequence[float]],
    scale: list[list[float | str]],
    title: str,
    value_fmt: Callable[[float], str] = lambda v: f'{v:.0%}',
    safety_mask: Sequence[Sequence[bool]] | None = None,
) -> str:
    """Heatmap. Builds a Vega-Lite spec rendered to SVG via vl-convert.

    ``cells[y][x]`` holds the value in [0, 1] for row ``y_labels[y]`` and
    column ``x_labels[x]``. Values < 0 mark absent cells and are rendered with
    the neutral grey sentinel ``#e4e2df`` (not the color scale). ``safety_mask[y][x]``
    marks a cell as a safety violation (rendered with a red stroke via Vega).
    Returns '' when labels are empty or vl-convert is unavailable.
    """
    if not x_labels or not y_labels:
        return ''
    from evaluatorq.common.reports.vega import render_svg, vl_heatmap

    cell_colors: list[list[str]] = []
    cell_texts: list[list[str]] = []
    for yi in range(len(y_labels)):
        color_row: list[str] = []
        text_row: list[str] = []
        for xi in range(len(x_labels)):
            value = float(cells[yi][xi])
            # value < 0 marks an absent cell -> neutral grey sentinel, not the scale.
            color = '#e4e2df' if value < 0 else scale_color(value, scale)
            color_row.append(color)
            text_row.append(value_fmt(value))
        cell_colors.append(color_row)
        cell_texts.append(text_row)
    safety: list[list[bool]] | None = None
    if safety_mask is not None:
        safety = [[bool(safety_mask[yi][xi]) for xi in range(len(x_labels))] for yi in range(len(y_labels))]
    spec = vl_heatmap(
        x_labels=list(x_labels),
        y_labels=list(y_labels),
        cell_colors=cell_colors,
        cell_texts=cell_texts,
        safety_mask=safety,
    )
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


def render_histogram(*, values: list[float], bins: int, title: str, width: int = 460, height: int = 220) -> str:
    """Histogram. Builds a Vega-Lite spec rendered to SVG via vl-convert.

    Includes count labels on bars and a dashed mean marker. Returns '' when
    *values* is empty, *bins* <= 0, or vl-convert is unavailable. The ``width``
    and ``height`` kwargs are accepted for API compatibility.
    """
    if not values or bins <= 0:
        return ''
    from evaluatorq.common.reports.vega import render_svg, vl_histogram

    mean = sum(float(v) for v in values) / len(values)
    spec = vl_histogram(values=values, bins=bins, mean=mean)
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


def render_line_chart(
    *,
    x_labels: list[str],
    series: list[tuple[str, list[float | None]]],
    title: str,
    width: int = 460,
    height: int = 220,
) -> str:
    """Multi-series line chart. Builds a Vega-Lite spec rendered to SVG via vl-convert.

    A ``None`` in a series is a gap (not measured) — the line breaks across it
    rather than dropping to zero. Each series gets a distinct strokeDash and
    point shape for colour-blind accessibility. Returns '' when labels or series
    are empty, or vl-convert is unavailable. The ``width`` and ``height`` kwargs
    are accepted for API compatibility.
    """
    if not x_labels or not series:
        return ''
    from evaluatorq.common.reports.vega import render_svg, vl_line

    spec = vl_line(x_labels=x_labels, series=series)
    svg = render_svg(spec)
    if not svg:
        return ''
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{svg}</figure>'


def render_sparkline(values: list[float], *, width: int = 80, height: int = 20) -> str:
    """Tiny inline mini-bar chart backed by Vega-Lite. Returns '' when empty.

    Builds a Vega-Lite spec rendered to SVG via vl-convert. The ``width`` and
    ``height`` kwargs are accepted for API compatibility.
    """
    if not values:
        return ''
    from evaluatorq.common.reports.vega import render_svg, vl_sparkline

    spec = vl_sparkline(values=values)
    return render_svg(spec)


def status_badge(text: str, status: str) -> str:
    """Semantic pill. ``status`` in {pass, fail, warn, neutral}."""
    safe_status = status if status in {'pass', 'fail', 'warn', 'neutral'} else 'neutral'
    return f'<span class="status-badge status-badge--{safe_status}">{esc(text)}</span>'


def kpi_cards(cards: list[dict[str, str]]) -> str:
    """Render a KPI scorecard band. Each card: {label, value, status?}."""
    if not cards:
        return ''
    items = []
    for c in cards:
        status = c.get('status', 'neutral')
        safe_status = status if status in {'pass', 'fail', 'warn', 'neutral'} else 'neutral'
        items.append(
            f'<div class="kpi-card kpi-card--{safe_status}">'
            f'<div class="kpi-value">{esc(c["value"])}</div>'
            f'<div class="kpi-label">{esc(c["label"])}</div></div>'
        )
    return f'<div class="kpi-band">{"".join(items)}</div>'
