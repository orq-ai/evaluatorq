"""HTML formatting helpers shared across report renderers.

Brand colors, the parameterized CSS loader, and small HTML primitives
(``esc``, ``html_table``, ``pct``, ``truncate``) live here. Chart helpers
that wrap Plotly + kaleido degrade gracefully when those packages are
unavailable. Hand-authored SVG/CSS chart primitives (``scale_color``,
``svg_donut``, ``svg_bar``, ``render_heatmap``, ``render_histogram``,
``render_line_chart``, ``render_sparkline``, ``kpi_cards``, ``status_badge``)
require no external dependencies.
"""

from __future__ import annotations

import html
import math
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
        parts.extend(f'<td>{cell}</td>' for cell in row)
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Chart helpers (Plotly + kaleido optional)
# ---------------------------------------------------------------------------


def charts_available() -> bool:
    """Check whether plotly and kaleido are importable."""
    try:
        import kaleido  # noqa: F401
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


def try_render_svg(fig: Any) -> str | None:
    """Render a Plotly figure as inline SVG, or ``None`` on failure."""
    try:
        svg_bytes = fig.to_image(format='svg', engine='kaleido')
        return svg_bytes.decode('utf-8') if isinstance(svg_bytes, bytes) else svg_bytes
    except Exception:
        logger.warning('Plotly SVG render failed (kaleido engine); chart omitted', exc_info=True)
        return None


def render_donut_chart(
    *,
    labels: list[str],
    values: list[int],
    colors: list[str],
    title: str,
) -> str:
    """Render a donut chart and wrap the SVG in a ``chart-container`` div.

    Returns an empty string when Plotly is unavailable, all values are zero,
    or the SVG render fails.
    """
    if not charts_available():
        return ''

    filtered = [(lbl, v, c) for lbl, v, c in zip(labels, values, colors, strict=False) if v > 0]
    if not filtered:
        return ''
    labels_f, values_f, colors_f = zip(*filtered, strict=False)

    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(labels_f),
                values=list(values_f),
                hole=0.5,
                marker=dict(colors=list(colors_f)),
                textinfo='label+percent',
                textfont=dict(size=12),
            )
        ]
    )
    fig.update_layout(
        width=400,
        height=300,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    svg = try_render_svg(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ''


def render_horizontal_bar_chart(
    *,
    labels: list[str],
    values: list[float],
    color: str,
    title: str,
    x_title: str,
    value_suffix: str = '',
) -> str:
    """Render a horizontal bar chart with values displayed outside the bars."""
    if not charts_available() or not labels:
        return ''

    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=values,
                orientation='h',
                marker_color=color,
                text=[f'{v:.0f}{value_suffix}' for v in values],
                textposition='outside',
            )
        ]
    )
    fig.update_layout(
        width=500,
        height=max(250, len(labels) * 35 + 80),
        margin=dict(t=40, b=40, l=80, r=50),
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_title,
        xaxis=dict(range=[0, max(max(values) * 1.2, 5) if values else 100]),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    svg = try_render_svg(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ''


# ---------------------------------------------------------------------------
# Hand-authored SVG/CSS chart primitives (no plotly / kaleido required)
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


def _arc_path(cx: float, cy: float, r_outer: float, r_inner: float, a0: float, a1: float) -> str:
    """SVG path for a donut segment between angles a0..a1 (radians, 0 = top, clockwise)."""

    def pt(r: float, a: float) -> tuple[float, float]:
        return cx + r * math.sin(a), cy - r * math.cos(a)

    large = 1 if (a1 - a0) > math.pi else 0
    x0o, y0o = pt(r_outer, a0)
    x1o, y1o = pt(r_outer, a1)
    x1i, y1i = pt(r_inner, a1)
    x0i, y0i = pt(r_inner, a0)
    return (
        f'M{x0o:.2f},{y0o:.2f} '
        f'A{r_outer:.2f},{r_outer:.2f} 0 {large} 1 {x1o:.2f},{y1o:.2f} '
        f'L{x1i:.2f},{y1i:.2f} '
        f'A{r_inner:.2f},{r_inner:.2f} 0 {large} 0 {x0i:.2f},{y0i:.2f} Z'
    )


def svg_donut(
    *,
    labels: list[str],
    values: list[float],
    colors: list[str],
    center_label: str,
    title: str,
    size: int = 220,
) -> str:
    """Hand-authored SVG donut. Returns '' when all values are zero."""
    total = sum(v for v in values if v > 0)
    if total <= 0:
        return ''
    cx = cy = size / 2
    r_outer = size / 2 - 4
    r_inner = r_outer * 0.62
    parts: list[str] = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="{esc(title)}">']
    angle = 0.0
    for label, value, color in zip(labels, values, colors, strict=False):
        if value <= 0:
            continue
        frac = value / total
        if frac >= 1.0:
            # A single slice covering the whole circle would produce a degenerate
            # arc (start == end), which SVG drops entirely. Emit a full annulus as
            # two half-arc paths so the ring stays visible.
            mid = angle + math.pi
            end = angle + 2 * math.pi
            parts.extend([
                (
                    f'<path d="{_arc_path(cx, cy, r_outer, r_inner, angle, mid)}" '
                    f'fill="{color}"><title>{esc(label)}: {value:g}</title></path>'
                ),
                (
                    f'<path d="{_arc_path(cx, cy, r_outer, r_inner, mid, end)}" '
                    f'fill="{color}"><title>{esc(label)}: {value:g}</title></path>'
                ),
            ])
            angle = end
            continue
        a1 = angle + frac * 2 * math.pi
        parts.append(
            f'<path d="{_arc_path(cx, cy, r_outer, r_inner, angle, a1)}" '
            f'fill="{color}"><title>{esc(label)}: {value:g}</title></path>'
        )
        angle = a1
    parts.extend([
        (
            f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
            f'class="donut-center">{esc(center_label)}</text>'
        ),
        '</svg>',
    ])
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


def svg_bar(
    *,
    rows: list[tuple[str, float]],
    title: str,
    color: str | None = None,
    width: int = 420,
    bar_height: int = 22,
    gap: int = 10,
) -> str:
    """Horizontal bar chart as hand-authored SVG. Returns '' when no rows."""
    if not rows:
        return ''
    from evaluatorq.common.reports.palette import COLORS as _COLORS

    bar_color = color or _COLORS['teal_400']
    label_w = 130
    max_val = max((v for _, v in rows), default=0) or 1
    plot_w = width - label_w - 48
    height = len(rows) * (bar_height + gap) + gap
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    y = gap
    for label, value in rows:
        bar_w = (value / max_val) * plot_w
        parts.extend([
            (
                f'<text x="{label_w - 8}" y="{y + bar_height / 2}" text-anchor="end" '
                f'dominant-baseline="central" class="bar-label">{esc(label)}</text>'
            ),
            (
                f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" '
                f'rx="3" fill="{bar_color}"></rect>'
            ),
            (
                f'<text x="{label_w + bar_w + 6:.1f}" y="{y + bar_height / 2}" '
                f'dominant-baseline="central" class="bar-value">{value:g}</text>'
            ),
        ])
        y += bar_height + gap
    parts.append('</svg>')
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


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
    """Heatmap as an HTML table of color-filled cells.

    ``cells[y][x]`` holds the value in [0, 1] for row ``y_labels[y]`` and
    column ``x_labels[x]``. ``safety_mask[y][x]`` marks a cell as a safety
    violation (rendered with the ``heatmap-cell--safety`` modifier).
    """
    if not x_labels or not y_labels:
        return ''
    head = ''.join(f'<th>{esc(x)}</th>' for x in x_labels)
    body_rows: list[str] = []
    for yi, ylabel in enumerate(y_labels):
        tds = ['<td class="heatmap-row-label"><strong>' + esc(ylabel) + '</strong></td>']
        for xi in range(len(x_labels)):
            value = float(cells[yi][xi])
            # value < 0 marks an absent cell -> neutral grey (matches --c-border), not the scale.
            color = '#e4e2df' if value < 0 else scale_color(value, scale)
            is_safety = bool(safety_mask and safety_mask[yi][xi])
            cls = 'heatmap-cell heatmap-cell--safety' if is_safety else 'heatmap-cell'
            tds.append(
                '<td><span class="'
                + cls
                + '" style="background:'
                + color
                + '">'
                + esc(value_fmt(value))
                + '</span></td>'
            )
        body_rows.append('<tr>' + ''.join(tds) + '</tr>')
    return (
        '<figure class="chart-card"><figcaption>' + esc(title) + '</figcaption>'
        '<div style="overflow-x:auto"><table class="heatmap-table">'
        '<thead><tr><th></th>' + head + '</tr></thead>'
        '<tbody>' + ''.join(body_rows) + '</tbody></table></div></figure>'
    )


def render_histogram(*, values: list[float], bins: int, title: str, width: int = 420, height: int = 180) -> str:
    """Histogram of values in [0, 1] as hand-authored SVG. Returns '' when empty."""
    if not values or bins <= 0:
        return ''
    from evaluatorq.common.reports.palette import COLORS as _COLORS

    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int(float(v) * bins)))
        counts[idx] += 1
    max_count = max(counts) or 1
    pad = 28
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    bar_w = plot_w / bins
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    for i, c in enumerate(counts):
        bar_h = (c / max_count) * plot_h
        x = pad + i * bar_w
        y = pad + (plot_h - bar_h)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" height="{bar_h:.1f}" '
            f'rx="2" fill="{_COLORS["teal_400"]}"><title>{i / bins:.1f}-{(i + 1) / bins:.1f}: {c}</title></rect>'
        )
        if c:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="bar-value">{c}</text>'
            )
    parts.append('</svg>')
    return f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>{"".join(parts)}</figure>'


def render_line_chart(
    *,
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    title: str,
    width: int = 460,
    height: int = 220,
) -> str:
    """Multi-series line chart (values in [0, 1]) as hand-authored SVG."""
    if not x_labels or not series:
        return ''
    from evaluatorq.common.reports.palette import QUALITATIVE

    pad = 32
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    n = max(1, len(x_labels) - 1)

    def x_at(i: int) -> float:
        return pad + (i / n) * plot_w

    def y_at(v: float) -> float:
        return pad + (1 - max(0.0, min(1.0, v))) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    # axes
    parts.extend([
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" class="axis"/>',
        f'<line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" class="axis"/>',
    ])
    legend = []
    for si, (name, ys) in enumerate(series):
        color = QUALITATIVE[si % len(QUALITATIVE)]
        pts = ' '.join(f'{x_at(i):.1f},{y_at(v):.1f}' for i, v in enumerate(ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend.append(
            f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>'
            f'{esc(name)}</span>'
        )
    parts.append('</svg>')
    return (
        f'<figure class="chart-card"><figcaption>{esc(title)}</figcaption>'
        f'{"".join(parts)}<div class="legend">{"".join(legend)}</div></figure>'
    )


def render_sparkline(values: list[float], *, width: int = 80, height: int = 20) -> str:
    """Tiny inline mini-bar SVG for table rows. Returns '' when empty."""
    if not values:
        return ''
    from evaluatorq.common.reports.palette import COLORS as _COLORS

    max_val = max(values) or 1
    bar_w = width / len(values)
    parts = [f'<svg class="sparkline" viewBox="0 0 {width} {height}" preserveAspectRatio="none">']
    for i, v in enumerate(values):
        bar_h = (v / max_val) * height
        parts.append(
            f'<rect x="{i * bar_w:.1f}" y="{height - bar_h:.1f}" '
            f'width="{bar_w - 1:.1f}" height="{bar_h:.1f}" fill="{_COLORS["teal_400"]}"/>'
        )
    parts.append('</svg>')
    return ''.join(parts)


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
