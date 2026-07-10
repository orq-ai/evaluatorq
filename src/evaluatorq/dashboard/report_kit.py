"""Shared design-component primitives for the aligned dashboard report pages.

Small pure functions emitting HTML strings, ``esc()`` on all dynamic text, no
state, no vl-convert. Built to serve both the Agent Sim report (this task) and
the Red Team report alignment that follows. Exact styling values live in the
matching spec (docs/superpowers/specs/2026-07-10-agent-sim-report-alignment-design.md)
and in the ``.sim-report`` CSS block in ``styles.py``.
"""

from __future__ import annotations

import itertools
import math
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import pct

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _best_worst_cells(heatmap_data: dict[str, Any]) -> tuple[dict | None, dict | None]:
    """Best/worst persona x scenario cells by success_rate, excluding cells with
    no scored (non-errored) conversations (``n == 0``)."""
    scored = [c for c in heatmap_data.get('cells', []) if c.get('n', 0) > 0]
    if not scored:
        return None, None
    best = max(scored, key=itemgetter('success_rate'))
    worst = min(scored, key=itemgetter('success_rate'))
    return best, worst


def _confidence_pill(confidence: str | None) -> str:
    if not confidence:
        return ''
    tone = {'HIGH': 'green-600', 'MEDIUM': 'amber-600', 'LOW': 'red-600'}.get(confidence.upper(), 'teal-600')
    return (
        f'<span class="es-confidence" style="border-color:var(--{tone});color:var(--{tone})">'
        f'{esc(confidence.upper())} CONFIDENCE</span>'
    )


def exec_summary(*, summary_data: dict[str, Any], heatmap_data: dict[str, Any], confidence: str | None) -> str:
    """Orange-left-bar executive-summary callout (spec §Overview.1)."""
    total = summary_data.get('total_conversations', 0)
    if not total:
        return ''
    personas = heatmap_data.get('personas', [])
    scenarios = heatmap_data.get('scenarios', [])
    success = pct(summary_data.get('success_rate', 0.0))
    avg_score = f'{summary_data.get("avg_goal_completion_score", 0.0):.2f}'

    sentence = (
        f'Across <strong>{total}</strong> conversations '
        f'({len(personas)} personas &times; {len(scenarios)} scenarios), the agent achieved a '
        f'<strong>{success}</strong> goal-completion rate at an average score of <strong>{avg_score}</strong>.'
    )
    best, worst = _best_worst_cells(heatmap_data)
    if best is not None and worst is not None and best is not worst:
        sentence += (
            f' It performed best on <strong>{esc(best["persona"])} &times; {esc(best["scenario"])}</strong> '
            f'({pct(best["success_rate"])}) and weakest on '
            f'<strong>{esc(worst["persona"])} &times; {esc(worst["scenario"])}</strong> ({pct(worst["success_rate"])}).'
        )

    return callout(sentence, confidence=confidence)


def callout(body_html: str, *, label: str = 'Executive summary', confidence: str | None = None) -> str:
    """Shared orange-left-bar callout shell (spec §Overview.1 / §report_kit generalizations).
    Sim's ``exec_summary`` builds its sentence then calls this; RT builds its own
    sentence in ``report_tabs.py`` and calls this directly."""
    return (
        '<div class="exec-summary">'
        '<div class="es-head">'
        f'<span class="es-label">{esc(label)}</span>'
        f'{_confidence_pill(confidence)}'
        '</div>'
        f'<p class="es-body">{body_html}</p>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------

_TAG_TONES: dict[str, str] = {
    'red': 'var(--red-600)',
    'green': 'var(--green-600)',
    'amber': 'var(--amber-600)',
    'teal': 'var(--teal-600)',
}


def tag(text: str, *, tone: str | None = None) -> str:
    """Inline summary pill, e.g. "3 turns" / "score 0.82" (spec §Transcripts)."""
    if not text:
        return ''
    style = ''
    color = _TAG_TONES.get(tone or '')
    if color:
        style = f' style="color:{color};border-color:{color}"'
    return f'<span class="rk-tag"{style}>{esc(text)}</span>'


def dial(value_text: str, fraction: float, *, radius: int, stroke: int, color: str, sub: str = '') -> str:
    """SVG gauge: sunken track + colored arc, ``esc()`` on ``value_text``/``sub``
    (spec §report_kit generalizations). ``fraction`` is clamped to [0, 1]."""
    size = (radius + stroke) * 2
    cx = cy = size / 2
    circ = 2 * math.pi * radius
    frac = max(0.0, min(1.0, fraction))
    dash = frac * circ
    sub_html = (
        f'<tspan x="{cx}" dy="1.2em" font-size="10" font-weight="500" fill="var(--text-faint)">{esc(sub)}</tspan>'
        if sub
        else ''
    )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="rk-dial">'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="var(--chart-track)" '
        f'stroke-width="{stroke}"></circle>'
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" '
        f'transform="rotate(-90 {cx} {cy})"></circle>'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" font-size="14" '
        f'font-weight="600" fill="var(--text-strong)">{esc(value_text)}{sub_html}</text>'
        '</svg>'
    )


_PILL_TONES: dict[str, dict[str, str]] = {
    'danger': {'bg': 'var(--red-100)', 'fg': 'var(--red-600)', 'dot': 'var(--red-600)'},
    'danger-solid': {'bg': 'var(--red-600)', 'fg': '#fff', 'dot': '#fff'},
    'warn': {'bg': 'var(--amber-100)', 'fg': 'var(--red-600)', 'dot': 'var(--orange-500)'},
    'good': {'bg': 'var(--green-100)', 'fg': 'var(--teal-600)', 'dot': 'var(--green-600)'},
    'neutral': {'bg': 'var(--sunken)', 'fg': 'var(--text-muted)', 'dot': 'var(--text-muted)'},
}


def pill(text: str, *, tone: str, dot: bool = False) -> str:
    """Small rounded status pill (spec §report_kit generalizations / Deviation #14).
    Tones: danger / danger-solid / warn / good / neutral. Optional leading dot."""
    colors = _PILL_TONES.get(tone, _PILL_TONES['neutral'])
    dot_html = f'<span class="rk-pill-dot" style="background:{colors["dot"]}"></span>' if dot else ''
    return f'<span class="rk-pill" style="background:{colors["bg"]};color:{colors["fg"]}">{dot_html}{esc(text)}</span>'


_SEVERITY_TONES: dict[str, str] = {'critical': 'danger-solid', 'high': 'warn', 'medium': 'neutral', 'low': 'good'}


def severity_pill(level: str) -> str:
    """SeverityBadge: critical->danger-solid, high->warn, medium->neutral, low->good."""
    key = (level or '').lower()
    return pill((level or '').capitalize(), tone=_SEVERITY_TONES.get(key, 'neutral'))


_OUTCOME_TONES: dict[str, tuple[str, str]] = {
    'vulnerable': ('danger', 'Vulnerable'),
    'resistant': ('good', 'Resistant'),
    'error': ('warn', 'Error'),
}


def outcome_pill(status: str) -> str:
    """OutcomeBadge (dotted pill): vulnerable->danger, resistant->good, error->warn."""
    tone, label = _OUTCOME_TONES.get((status or '').lower(), ('neutral', status or ''))
    return pill(label, tone=tone, dot=True)


def donut(
    segments: list[dict[str, Any]],
    center_value: str,
    center_label: str,
) -> str:
    """Ring SVG + center text + legend list (extracted from the sim outcomes
    donut per spec §Shared infrastructure #3). Each segment is
    ``{'label': str, 'value': number, 'color': str}``. Callers own the
    surrounding ``<figure>``/``panel()`` wrapper and caption."""
    total = sum(s.get('value', 0) for s in segments)
    radius = 60
    circ = 2 * math.pi * radius
    arcs: list[str] = []
    offset = 0.0
    for s in segments:
        value = s.get('value', 0)
        if value <= 0 or total <= 0:
            continue
        length = circ * value / total
        arcs.append(
            f'<circle cx="75" cy="75" r="{radius}" fill="none" stroke="{s.get("color", "")}" stroke-width="18"'
            f' stroke-dasharray="{length:.1f} {circ - length:.1f}" stroke-dashoffset="{-offset:.1f}"/>'
        )
        offset += length
    legend = ''.join(
        f'<li><span class="donut-key" style="background:{s.get("color", "")}"></span>'
        f'{esc(s.get("label", ""))} · {s.get("value", 0)}</li>'
        for s in segments
        if s.get('value', 0) > 0
    )
    return (
        '<div class="donut-wrap"><div class="donut">'
        f'<svg width="150" height="150" viewBox="0 0 150 150">{"".join(arcs)}</svg>'
        f'<div class="donut-center"><span class="donut-value">{esc(center_value)}</span>'
        f'<span class="donut-label">{esc(center_label)}</span></div></div>'
        f'<ul class="donut-legend">{legend}</ul></div>'
    )


def panel(title: str, body: str, *, sub: str | None = None) -> str:
    """White card wrapper with a mono-label title and pre-rendered ``body`` HTML."""
    if not title:
        return ''
    sub_html = f'<div class="rk-panel-sub">{esc(sub)}</div>' if sub else ''
    return (
        '<div class="rk-panel">'
        f'<div class="rk-panel-title">{esc(title)}</div>'
        f'{sub_html}'
        f'<div class="rk-panel-body">{body}</div>'
        '</div>'
    )


def bar_rows(
    rows: list[tuple[str, float]],
    *,
    width: int,
    label_w: int,
    color: str = 'var(--chart-2)',
    fmt: Callable[[float], str] = str,
    color_scale: Sequence[tuple[float, str]] | None = None,
    max_value: float | None = None,
) -> str:
    """Horizontal SVG bar rows; bar length proportional to value/max (spec §Overview.3 / §Breakdown.4 / §Turn.3).

    ``color_scale``, when given, interpolates each bar's fill via
    ``_interp_color(value / max_v, stops=color_scale)`` instead of the flat
    ``color``. ``max_value``, when given, overrides the computed
    ``max(values)`` denominator used for bar scaling.
    """
    if not rows:
        return ''
    computed_max = max((v for _, v in rows), default=0.0) or 1.0
    max_v = max_value if max_value is not None else computed_max
    row_h = 28
    bar_h = 14
    track_w = max(width - label_w - 56, 10)
    height = len(rows) * row_h
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" class="rk-bar-rows">']
    for i, (label, value) in enumerate(rows):
        y = i * row_h
        cy = y + row_h / 2
        bar_w = (value / max_v) * track_w if max_v else 0.0
        fill = _interp_color(value / max_v, stops=color_scale) if color_scale else color
        parts.extend((
            (
                f'<text x="{label_w - 8}" y="{cy:.1f}" text-anchor="end" dominant-baseline="middle" '
                f'font-size="12" fill="var(--text-body)">{esc(label)}</text>'
            ),
            (
                f'<rect x="{label_w}" y="{y + (row_h - bar_h) / 2:.1f}" width="{bar_w:.1f}" height="{bar_h}" '
                f'rx="3" fill="{fill}"></rect>'
            ),
            (
                f'<text x="{label_w + bar_w + 8:.1f}" y="{cy:.1f}" dominant-baseline="middle" '
                f'font-family="var(--font-mono)" font-size="11" font-weight="600" fill="var(--text-strong)">'
                f'{esc(fmt(value))}</text>'
            ),
        ))
    parts.append('</svg>')
    return ''.join(parts)


def histogram(values: list[float], *, bins: int = 10, mean_line: bool = True) -> str:
    """Pure-SVG score-distribution histogram with a dashed mean line (spec §Breakdown.2)."""
    if not values:
        return ''
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    width, height = 640, 200
    pad_left, pad_right, pad_top, pad_bottom = 32, 10, 10, 20
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    bin_span = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / bin_span), bins - 1) if bin_span else 0
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_gap = 2
    bar_w = plot_w / bins - bar_gap

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" class="rk-histogram">']
    for i in range(3):
        gy = pad_top + plot_h * i / 2
        parts.append(
            f'<line x1="{pad_left}" y1="{gy:.1f}" x2="{width - pad_right}" y2="{gy:.1f}" '
            f'stroke="var(--chart-grid)" stroke-width="1"></line>'
        )
    for i, count in enumerate(counts):
        bar_h = plot_h * (count / max_count)
        x = pad_left + i * (plot_w / bins) + bar_gap / 2
        y = pad_top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="var(--chart-2)"></rect>'
        )
    parts.extend((
        (
            f'<text x="{pad_left}" y="{height - 4}" font-family="var(--font-mono)" font-size="10" '
            f'fill="var(--text-faint)">{esc(f"{lo:.2f}")}</text>'
        ),
        (
            f'<text x="{width - pad_right}" y="{height - 4}" text-anchor="end" font-family="var(--font-mono)" '
            f'font-size="10" fill="var(--text-faint)">{esc(f"{hi:.2f}")}</text>'
        ),
    ))
    if mean_line:
        mean = sum(values) / len(values)
        mx = pad_left + ((mean - lo) / (hi - lo)) * plot_w if hi != lo else pad_left + plot_w / 2
        parts.append(
            f'<line x1="{mx:.1f}" y1="{pad_top}" x2="{mx:.1f}" y2="{pad_top + plot_h}" '
            f'stroke="var(--orange-500)" stroke-width="1.5" stroke-dasharray="4 3"></line>'
        )
    parts.append('</svg>')
    return ''.join(parts)


_LINE_PALETTE: tuple[str, ...] = ('var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)')


def line_chart(turns: list[int], series: dict[str, list[float | None]]) -> str:
    """Multi-series SVG line chart with dots and a legend (spec §Turn.2). Empty ``series`` -> ``''``."""
    if not series:
        return ''
    width, height = 720, 260
    pad_left, pad_right, pad_top, pad_bottom = 36, 16, 16, 28
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    n = len(turns)

    def x_at(i: int) -> float:
        return pad_left + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def y_at(v: float) -> float:
        return pad_top + plot_h * (1 - max(0.0, min(1.0, v)))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" class="rk-line-chart">']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_top + plot_h * (1 - frac)
        parts.extend((
            (
                f'<line x1="{pad_left}" y1="{gy:.1f}" x2="{width - pad_right}" y2="{gy:.1f}" '
                f'stroke="var(--chart-grid)" stroke-width="1"></line>'
            ),
            (
                f'<text x="{pad_left - 6}" y="{gy + 3:.1f}" text-anchor="end" font-family="var(--font-mono)" '
                f'font-size="10" fill="var(--text-faint)">{frac:.2f}</text>'
            ),
        ))
    for i, t in enumerate(turns):
        parts.append(
            f'<text x="{x_at(i):.1f}" y="{height - pad_bottom + 16}" text-anchor="middle" '
            f'font-family="var(--font-mono)" font-size="10" fill="var(--text-faint)">{esc(str(t))}</text>'
        )

    legend_items: list[str] = []
    for idx, (name, values) in enumerate(series.items()):
        color = _LINE_PALETTE[idx % len(_LINE_PALETTE)]
        run: list[tuple[float, float]] = []
        for i, v in enumerate(values):
            if v is None:
                if len(run) > 1:
                    points = ' '.join(f'{px:.1f},{py:.1f}' for px, py in run)
                    parts.append(
                        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"></polyline>'
                    )
                run = []
                continue
            run.append((x_at(i), y_at(v)))
        if len(run) > 1:
            points = ' '.join(f'{px:.1f},{py:.1f}' for px, py in run)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"></polyline>')
        for i, v in enumerate(values):
            if v is None:
                continue
            px, py = x_at(i), y_at(v)
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="white" stroke="{color}" stroke-width="1.5"></circle>'
            )
        if any(v is not None for v in values):
            legend_items.append(
                f'<span class="rk-legend-item"><span class="rk-legend-swatch" style="background:{color}"></span>'
                f'{esc(name)}</span>'
            )
    parts.append('</svg>')
    legend_html = f'<div class="rk-legend">{"".join(legend_items)}</div>'
    return ''.join(parts) + legend_html


_HEAT_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (0xDF, 0x53, 0x25)),
    (0.5, (0xFF, 0x8F, 0x34)),
    (1.0, (0x29, 0x9D, 0x8F)),
)

# RT heat scale: "higher = worse" (low → light neutral, high → red).
SCALE_HEAT_RT: tuple[tuple[float, str], ...] = (
    (0.0, '#f3f1ee'),
    (0.5, '#ff8f34'),
    (1.0, '#df5325'),
)


def _to_rgb(c: tuple[int, int, int] | str) -> tuple[int, int, int]:
    """Normalize a stop color (int-triple or hex string) to an RGB int-triple."""
    if isinstance(c, str):
        h = c.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return c


def _interp_color(v: float, *, stops: Any = None) -> str:
    """Piecewise-linear RGB interpolation over a 3-stop scale (sim default when
    ``stops`` is not passed). Accepts either int-triple stops (``_HEAT_STOPS``)
    or hex-string stops (``SCALE_HEAT_RT``)."""
    v = 0.0 if v < 0 else 1.0 if v > 1 else v
    raw = stops if stops is not None else _HEAT_STOPS
    norm = [(pos, _to_rgb(c)) for pos, c in raw]
    for (lo, c0), (hi, c1) in itertools.pairwise(norm):
        if lo <= v <= hi:
            t = 0.0 if hi == lo else (v - lo) / (hi - lo)
            r, g, b = (round(a + (b_ - a) * t) for a, b_ in zip(c0, c1, strict=True))
            return f'#{r:02x}{g:02x}{b:02x}'
    last = norm[-1][1]
    return f'#{last[0]:02x}{last[1]:02x}{last[2]:02x}'


def heatmap(
    row_labels: list[str],
    col_labels: list[str],
    cells: list[dict],
    *,
    row_key: str = 'persona',
    col_key: str = 'scenario',
    value_key: str = 'success_rate',
    color_scale: Sequence[tuple[float, str]] | None = None,
) -> str:
    """HTML-table row x column heatmap (spec §Breakdown.1). Missing cells render sunken with "—".

    Sim defaults (``row_key='persona'``, ``col_key='scenario'``,
    ``value_key='success_rate'``, ``color_scale=None``) keep existing sim
    call sites and rendering unchanged.
    """
    lookup = {(c[row_key], c[col_key]): c for c in cells}
    header = ''.join(f'<th class="rk-heat-col">{esc(s)}</th>' for s in col_labels)
    body_rows: list[str] = []
    for p in row_labels:
        tds: list[str] = []
        for s in col_labels:
            cell = lookup.get((p, s))
            if cell is None:
                tds.append('<td class="rk-heat-cell rk-heat-empty">—</td>')
                continue
            rate = cell.get(value_key, 0.0)
            bg = _interp_color(rate, stops=color_scale)
            ink = 'var(--ink-900)' if rate <= 0.55 else '#fff'
            tds.append(f'<td class="rk-heat-cell" style="background:{bg};color:{ink}">{pct(rate)}</td>')
        body_rows.append(f'<tr><th class="rk-heat-row">{esc(p)}</th>{"".join(tds)}</tr>')
    return (
        '<table class="rk-heatmap"><thead><tr><th></th>'
        f'{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    )


def meta_grid(items: list[tuple[str, str | None]]) -> str:
    """Key/value metadata grid, ``repeat(auto-fill, minmax(190px,1fr))`` (spec §Config.1). Null/empty skipped."""
    rows = [(k, v) for k, v in items if v not in (None, '')]
    if not rows:
        return ''
    cells = ''.join(
        f'<div class="rk-meta-item"><div class="rk-meta-key">{esc(k)}</div>'
        f'<div class="rk-meta-value">{esc(v)}</div></div>'
        for k, v in rows
    )
    return f'<div class="rk-meta-grid">{cells}</div>'
