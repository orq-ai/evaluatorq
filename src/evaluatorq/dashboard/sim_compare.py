"""Side-by-side comparison of two simulation runs (RES-1085).

The single-run sim views live in ``sim_views.py``; this module adds a compare
surface on top of the same ``sim`` adapter/library. Nothing here touches the sim
runner or data model — it only reads two already-stored ``SimulationRun`` objects.

Routes (registered by ``register_sim_compare_routes``):

    GET /compare/sim?a={ridA}&b={ridB}
        Full page: header for both runs, a KPI delta table (aggregates +
        per-scorer + terminated-by), and an outcome-diff table matching
        conversations by ``(persona, scenario)``.

    GET /compare/sim/transcript?a={ridA}&b={ridB}&ia={idxA}&ib={idxB}
        Side-by-side transcript fragment for one matched conversation pair.

Conversations are matched by ``(persona, scenario)`` (the ticket's chosen key):
runs that share a dataset line up; anything unmatched is listed separately so a
mismatched dataset is visible rather than silently dropped.
"""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import kpi_cards, pct
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.sim_views import _load_run, render_transcript_fragment

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.simulation.types import SimulationEntry, SimulationRun


# ---------------------------------------------------------------------------
# KPI deltas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KpiRow:
    """One comparison row: a label, the two run values, and their delta.

    ``kind`` drives formatting: ``'rate'`` (0..1 → %), ``'score'`` (2dp),
    ``'turns'`` (1dp), ``'count'`` (int). ``a``/``b`` are floats; a metric
    absent from one run is treated as 0.0 so the delta is still meaningful.
    """

    label: str
    a: float
    b: float
    kind: str

    @property
    def delta(self) -> float:
        return self.b - self.a


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_kpis(
    entries_a: list[SimulationEntry], entries_b: list[SimulationEntry], run_a: SimulationRun, run_b: SimulationRun
) -> list[KpiRow]:
    """Build the KPI delta rows: aggregates, per-scorer averages, terminated-by.

    Aggregates are computed from the entry lists (so they track the conversations
    actually present); per-scorer values come from each run's ``scorer_averages``.
    """
    rows: list[KpiRow] = [
        KpiRow('Conversations', float(len(entries_a)), float(len(entries_b)), 'count'),
        KpiRow(
            'Goal-achieved rate',
            _mean([1.0 if e.goal_achieved else 0.0 for e in entries_a]),
            _mean([1.0 if e.goal_achieved else 0.0 for e in entries_b]),
            'rate',
        ),
        KpiRow(
            'Mean goal score',
            _mean([e.goal_completion_score for e in entries_a]),
            _mean([e.goal_completion_score for e in entries_b]),
            'score',
        ),
        KpiRow(
            'Mean turns',
            _mean([float(e.turn_count) for e in entries_a]),
            _mean([float(e.turn_count) for e in entries_b]),
            'turns',
        ),
    ]

    # Per-scorer averages — union of both runs' scorer_averages keys, sorted.
    sa_a = run_a.scorer_averages or {}
    sa_b = run_b.scorer_averages or {}
    rows.extend(
        KpiRow(f'Scorer: {key}', float(sa_a.get(key, 0.0)), float(sa_b.get(key, 0.0)), 'score')
        for key in sorted(set(sa_a) | set(sa_b))
    )

    # Terminated-by distribution — counts per reason, union of labels.
    tb_a = Counter(e.terminated_by for e in entries_a)
    tb_b = Counter(e.terminated_by for e in entries_b)
    rows.extend(
        KpiRow(f'Terminated: {label}', float(tb_a[label]), float(tb_b[label]), 'count')
        for label in sorted(set(tb_a) | set(tb_b))
    )

    return rows


def _fmt(value: float, kind: str) -> str:
    if kind == 'rate':
        return f'{value * 100:.0f}%'
    if kind == 'score':
        return f'{value:.2f}'
    if kind == 'turns':
        return f'{value:.1f}'
    return f'{value:.0f}'


def _fmt_delta(row: KpiRow) -> str:
    d = row.delta
    if abs(d) < 1e-9:
        return '<span class="cmp-delta cmp-flat">0</span>'
    arrow = '&#x25B2;' if d > 0 else '&#x25BC;'  # ▲ / ▼
    cls = 'cmp-up' if d > 0 else 'cmp-down'
    # Delta is formatted like its metric; counts render as a signed integer.
    if row.kind == 'rate':
        body = f'{d * 100:+.0f}%'
    elif row.kind == 'score':
        body = f'{d:+.2f}'
    elif row.kind == 'turns':
        body = f'{d:+.1f}'
    else:
        body = f'{d:+.0f}'
    return f'<span class="cmp-delta {cls}">{arrow} {body}</span>'


# ---------------------------------------------------------------------------
# Conversation matching (by persona + scenario)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedPair:
    a: SimulationEntry
    b: SimulationEntry


@dataclass(frozen=True)
class Matching:
    matched: list[MatchedPair]
    a_only: list[SimulationEntry]
    b_only: list[SimulationEntry]


def match_entries(entries_a: list[SimulationEntry], entries_b: list[SimulationEntry]) -> Matching:
    """Pair conversations across runs by ``(persona, scenario)``.

    Duplicate keys within a run are paired positionally (first A with first B,
    etc.); any leftovers on either side fall into ``a_only`` / ``b_only``.
    """
    by_key_b: dict[tuple[str, str], list[SimulationEntry]] = defaultdict(list)
    for e in entries_b:
        by_key_b[e.persona, e.scenario].append(e)

    matched: list[MatchedPair] = []
    a_only: list[SimulationEntry] = []
    consumed_b: set[int] = set()
    # Track position within each key's B-list as we consume duplicates.
    cursor: dict[tuple[str, str], int] = defaultdict(int)

    for ea in entries_a:
        key = (ea.persona, ea.scenario)
        bucket = by_key_b.get(key, [])
        i = cursor[key]
        if i < len(bucket):
            eb = bucket[i]
            cursor[key] = i + 1
            consumed_b.add(id(eb))
            matched.append(MatchedPair(ea, eb))
        else:
            a_only.append(ea)

    b_only = [e for e in entries_b if id(e) not in consumed_b]
    return Matching(matched, a_only, b_only)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _entries(run: SimulationRun) -> list[SimulationEntry]:
    from evaluatorq.simulation.reports.sections import individual_entries

    return individual_entries(run.results)


def _panel(title: str, sub: str, inner: str) -> str:
    """Framed section matching the dashboard's ``.panel`` chrome (theme-styled)."""
    return (
        f'<div class="panel"><div class="panel-title">{esc(title)}</div>'
        f'<div class="panel-sub">{esc(sub)}</div>{inner}</div>'
    )


def _agg(entries: list[SimulationEntry]) -> tuple[float, float, float]:
    """(goal-achieved rate, mean goal score, mean turns) for an entry list."""
    return (
        _mean([1.0 if e.goal_achieved else 0.0 for e in entries]),
        _mean([e.goal_completion_score for e in entries]),
        _mean([float(e.turn_count) for e in entries]),
    )


def _dir_status(delta: float, *, higher_better: bool) -> str:
    """Map a delta to a kpi-card status colour (pass/fail/neutral)."""
    if abs(delta) < 1e-9:
        return 'neutral'
    good = delta > 0 if higher_better else delta < 0
    return 'pass' if good else 'fail'


def _kpi_band(entries_a: list[SimulationEntry], entries_b: list[SimulationEntry]) -> str:
    """Headline KPI cards: current (B) value with the delta vs A in the label."""
    rate_a, score_a, turns_a = _agg(entries_a)
    rate_b, score_b, turns_b = _agg(entries_b)
    cards = [
        {
            'label': f'Goal-achieved · {rate_b - rate_a:+.0%} vs A',
            'value': pct(rate_b),
            'status': _dir_status(rate_b - rate_a, higher_better=True),
        },
        {
            'label': f'Mean score · {score_b - score_a:+.2f} vs A',
            'value': f'{score_b:.2f}',
            'status': _dir_status(score_b - score_a, higher_better=True),
        },
        {'label': f'Mean turns · {turns_b - turns_a:+.1f} vs A', 'value': f'{turns_b:.1f}', 'status': 'neutral'},
        {
            'label': f'Conversations · {len(entries_a)} → {len(entries_b)}',
            'value': str(len(entries_b)),
            'status': 'neutral',
        },
    ]
    return kpi_cards(cards)


# --- Inline-SVG charts --------------------------------------------------------
# Server-rendered SVG (no JS, no /static dependency) so charts always render — in
# the dashboard, in tests, and in static HTML exports alike. Colours are the orq
# brand palette; A is green, B is rust (the brand accent).
_SERIES_A = '#0f9d6b'
_SERIES_B = '#df5325'
_AXIS = '#d8d3cc'
_LBL = '#6b7280'
_VAL = '#25232e'
_MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace'


def _svg_grouped_bars(categories: list[str], series: list[tuple[str, str, list[float]]], *, value_kind: str) -> str:
    """Grouped horizontal bar SVG. ``series`` = [(name, colour, values)];
    ``value_kind`` is ``'rate'`` (0..1 → %) or ``'count'`` (integer)."""
    if not categories or not series:
        return ''
    all_vals = [v for _, _, vs in series for v in vs]
    vmax = 1.0 if value_kind == 'rate' else (max(all_vals) or 1.0)
    n = len(series)
    bar_h, bar_gap, group_gap = 12, 3, 14
    lbl_w, plot_x, plot_w, top, right_pad = 150, 158, 250, 34, 52
    group_h = n * (bar_h + bar_gap) + group_gap
    height = top + len(categories) * group_h + 8
    width = plot_x + plot_w + right_pad

    def fmt(v: float) -> str:
        return f'{v * 100:.0f}%' if value_kind == 'rate' else f'{v:.0f}'

    p: list[str] = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" style="max-width:{width}px">']
    lx = plot_x
    for name, color, _ in series:
        p.append(
            f'<rect x="{lx}" y="12" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{lx + 15}" y="21" font-family="{_MONO}" font-size="11" fill="{_LBL}">{esc(name)}</text>'
        )
        lx += 15 + int(len(name) * 6.7) + 18
    p.append(f'<line x1="{plot_x}" y1="{top}" x2="{plot_x}" y2="{height - 8}" stroke="{_AXIS}"/>')
    for ci, cat in enumerate(categories):
        gy = top + ci * group_h
        cy = gy + n * (bar_h + bar_gap) / 2
        p.append(
            f'<text x="{lbl_w - 8}" y="{cy + 3:.0f}" text-anchor="end" font-family="{_MONO}" '
            f'font-size="11" fill="{_LBL}">{esc(cat)}</text>'
        )
        for si, (_, color, vs) in enumerate(series):
            v = vs[ci] if ci < len(vs) else 0.0
            bw = max(0.0, v / vmax * plot_w)
            by = gy + si * (bar_h + bar_gap)
            p.append(
                f'<rect x="{plot_x}" y="{by}" width="{bw:.1f}" height="{bar_h}" rx="2" fill="{color}"/>'
                f'<text x="{plot_x + bw + 5:.1f}" y="{by + bar_h - 2}" font-family="{_MONO}" '
                f'font-size="10" fill="{_VAL}">{fmt(v)}</text>'
            )
    p.append('</svg>')
    return ''.join(p)


def _svg_diverging(rows: list[tuple[str, float]]) -> str:
    """Diverging horizontal bars around a zero line (positive right/green,
    negative left/rust). ``rows`` = [(label, delta)]."""
    if not rows:
        return ''
    vmax = max((abs(d) for _, d in rows), default=0.0) or 1.0
    bar_h, gap = 13, 6
    lbl_w, half, right_pad, top = 160, 120, 46, 12
    mid_x = lbl_w + half
    width = lbl_w + half * 2 + right_pad
    height = top + len(rows) * (bar_h + gap) + 6
    p: list[str] = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" style="max-width:{width}px">']
    p.append(f'<line x1="{mid_x}" y1="{top - 4}" x2="{mid_x}" y2="{height - 6}" stroke="{_AXIS}"/>')
    for i, (label, d) in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = abs(d) / vmax * half
        if d >= 0:
            x, color, tx, anchor = mid_x, _SERIES_A, mid_x + w + 5, 'start'
        else:
            x, color, tx, anchor = mid_x - w, _SERIES_B, mid_x - w - 5, 'end'
        p.append(
            f'<text x="{lbl_w - 8}" y="{y + bar_h - 2}" text-anchor="end" font-family="{_MONO}" '
            f'font-size="10" fill="{_LBL}">{esc(label)}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="2" fill="{color}"/>'
            f'<text x="{tx:.1f}" y="{y + bar_h - 2}" text-anchor="{anchor}" font-family="{_MONO}" '
            f'font-size="10" fill="{_VAL}">{d:+.2f}</text>'
        )
    p.append('</svg>')
    return ''.join(p)


def _outcomes_chart(
    entries_a: list[SimulationEntry], entries_b: list[SimulationEntry], name_a: str, name_b: str
) -> str:
    """A-vs-B bars for the two headline 0..1 metrics."""
    rate_a, score_a, _ = _agg(entries_a)
    rate_b, score_b, _ = _agg(entries_b)
    svg = _svg_grouped_bars(
        ['Goal-achieved rate', 'Mean goal score'],
        [(name_a, _SERIES_A, [rate_a, score_a]), (name_b, _SERIES_B, [rate_b, score_b])],
        value_kind='rate',
    )
    return _panel('Outcomes', 'Goal rate and mean score · A vs B', svg)


def _scorer_chart(run_a: SimulationRun, run_b: SimulationRun, name_a: str, name_b: str) -> str:
    """A-vs-B bars over the union of both runs' scorer averages."""
    sa_a = run_a.scorer_averages or {}
    sa_b = run_b.scorer_averages or {}
    keys = sorted(set(sa_a) | set(sa_b))
    if not keys:
        return ''
    svg = _svg_grouped_bars(
        keys,
        [(name_a, _SERIES_A, [sa_a.get(k, 0.0) for k in keys]), (name_b, _SERIES_B, [sa_b.get(k, 0.0) for k in keys])],
        value_kind='rate',
    )
    return _panel('Scorer averages', 'Per-scorer mean · A vs B', svg)


def _terminated_chart(
    entries_a: list[SimulationEntry], entries_b: list[SimulationEntry], name_a: str, name_b: str
) -> str:
    """A-vs-B bars for how conversations ended."""
    tb_a = Counter(e.terminated_by for e in entries_a)
    tb_b = Counter(e.terminated_by for e in entries_b)
    labels = sorted(set(tb_a) | set(tb_b))
    if not labels:
        return ''
    svg = _svg_grouped_bars(
        labels,
        [(name_a, _SERIES_A, [float(tb_a[x]) for x in labels]), (name_b, _SERIES_B, [float(tb_b[x]) for x in labels])],
        value_kind='count',
    )
    return _panel('How conversations ended', 'Terminated-by counts · A vs B', svg)


def _delta_chart(matching: Matching) -> str:
    """Per-conversation goal-score change (B - A) over matched pairs, sorted."""
    if not matching.matched:
        return ''
    rows = sorted(
        (
            (f'{pair.a.persona}/{pair.a.scenario}', pair.b.goal_completion_score - pair.a.goal_completion_score)
            for pair in matching.matched
        ),
        key=operator.itemgetter(1),
    )
    return _panel(
        'Per-conversation score change',
        'B minus A on matched conversations · green improved, rust regressed',
        _svg_diverging(rows),
    )


def _all_metrics_table(rows: list[KpiRow], name_a: str, name_b: str) -> str:
    """Exhaustive metric table: every KPI row with A, B, and the delta."""
    body = ''.join(
        f'<tr><td>{esc(r.label)}</td>'
        f'<td>{_fmt(r.a, r.kind)}</td>'
        f'<td>{_fmt(r.b, r.kind)}</td>'
        f'<td>{_fmt_delta(r)}</td></tr>'
        for r in rows
    )
    table = (
        '<table class="cmp-table"><thead><tr>'
        f'<th>Metric</th><th>{esc(name_a)}</th><th>{esc(name_b)}</th><th>&Delta; (B - A)</th>'
        f'</tr></thead><tbody>{body}</tbody></table>'
    )
    return _panel('All metrics', 'Full A / B / delta breakdown', table)


def _diff_table(rid_a: str, rid_b: str, matching: Matching) -> str:
    safe_a, safe_b = esc(rid_a), esc(rid_b)
    rows: list[str] = []
    for pair in matching.matched:
        a, b = pair.a, pair.b
        ga = 'yes' if a.goal_achieved else 'no'
        gb = 'yes' if b.goal_achieved else 'no'
        flip = a.goal_achieved != b.goal_achieved
        flip_html = '<span class="cmp-flip">flip</span>' if flip else ''
        d_score = b.goal_completion_score - a.goal_completion_score
        d_turns = b.turn_count - a.turn_count
        rows.append(
            f'<tr class="cmp-diff-row" style="cursor:pointer"'
            f' hx-get="/compare/sim/transcript?a={safe_a}&b={safe_b}&ia={a.index}&ib={b.index}"'
            f' hx-target="#cmp-transcript-panel" hx-swap="innerHTML">'
            f'<td>{esc(a.persona)}</td><td>{esc(a.scenario)}</td>'
            f'<td>{ga} &rarr; {gb} {flip_html}</td>'
            f'<td>{a.goal_completion_score:.2f} &rarr; {b.goal_completion_score:.2f} ({d_score:+.2f})</td>'
            f'<td>{a.turn_count} &rarr; {b.turn_count} ({d_turns:+d})</td>'
            f'</tr>'
        )
    matched_html = (
        '<table class="cmp-table"><thead><tr>'
        '<th>Persona</th><th>Scenario</th><th>Goal (A&rarr;B)</th>'
        '<th>Score (A&rarr;B)</th><th>Turns (A&rarr;B)</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        if rows
        else '<p class="sim-empty">No conversations matched on (persona, scenario).</p>'
    )

    def _unmatched(title: str, entries: list[SimulationEntry]) -> str:
        if not entries:
            return ''
        items = ''.join(f'<li>{esc(e.persona)} &middot; {esc(e.scenario)}</li>' for e in entries)
        return f'<div class="cmp-unmatched"><strong>{esc(title)}</strong><ul>{items}</ul></div>'

    unmatched = _unmatched('Only in A', matching.a_only) + _unmatched('Only in B', matching.b_only)

    transcript_slot = (
        '<div id="cmp-transcript-panel" class="cmp-transcript-panel">'
        '<p class="sim-select-prompt">Select a matched conversation above to compare transcripts.</p></div>'
    )
    return _panel(
        'Outcome diffs',
        'Matched by persona and scenario · click a row to compare transcripts',
        matched_html + unmatched + transcript_slot,
    )


def render_compare_transcript(entry_a: SimulationEntry, entry_b: SimulationEntry, name_a: str, name_b: str) -> str:
    """Two single-run transcript fragments laid out side by side."""
    return (
        '<div class="cmp-transcript-grid">'
        f'<div class="cmp-side"><h3 class="cmp-side-title">{esc(name_a)}</h3>{render_transcript_fragment(entry_a)}</div>'
        f'<div class="cmp-side"><h3 class="cmp-side-title">{esc(name_b)}</h3>{render_transcript_fragment(entry_b)}</div>'
        '</div>'
    )


def _hero(run_a: SimulationRun, run_b: SimulationRun) -> str:
    """Editorial header: mono kicker, display headline, run names + dates."""
    date_a = run_a.created_at.strftime('%Y-%m-%d %H:%M')
    date_b = run_b.created_at.strftime('%Y-%m-%d %H:%M')
    return (
        '<div class="report-head"><a class="report-back" href="/?surface=sim">&larr; Agent sim runs</a></div>'
        '<section class="cmp-hero">'
        '<div class="cmp-kicker">Agent sim // run comparison</div>'
        f'<h1 class="cmp-title">{esc(run_a.run_name)} <span class="cmp-vs">vs</span> {esc(run_b.run_name)}</h1>'
        f'<p class="cmp-sub">A: {esc(run_a.run_name)} · {date_a} &nbsp;&nbsp; B: {esc(run_b.run_name)} · {date_b}</p>'
        '</section>'
    )


def render_compare_page(rid_a: str, rid_b: str, run_a: SimulationRun, run_b: SimulationRun) -> str:
    """Full compare-page body: hero, KPI deltas + charts, all-metrics, diffs."""
    entries_a = _entries(run_a)
    entries_b = _entries(run_b)
    kpis = compare_kpis(entries_a, entries_b, run_a, run_b)
    matching = match_entries(entries_a, entries_b)
    name_a, name_b = run_a.run_name, run_b.run_name

    kpi_section = _panel('KPI deltas', 'Headline metrics, B relative to A', _kpi_band(entries_a, entries_b))
    charts = (
        '<div class="cmp-charts">'
        f'{_outcomes_chart(entries_a, entries_b, name_a, name_b)}'
        f'{_scorer_chart(run_a, run_b, name_a, name_b)}'
        f'{_terminated_chart(entries_a, entries_b, name_a, name_b)}'
        '</div>'
    )
    return (
        f'{_COMPARE_CSS}{_hero(run_a, run_b)}'
        f'{kpi_section}{charts}'
        f'{_delta_chart(matching)}'
        f'{_all_metrics_table(kpis, name_a, name_b)}'
        f'{_diff_table(rid_a, rid_b, matching)}'
    )


_COMPARE_CSS = (
    '<style>'
    '.cmp-hero { margin: 4px 0 18px; }'
    '.cmp-kicker { font-family: var(--font-mono); font-size: 11px; letter-spacing: .14em;'
    ' text-transform: uppercase; color: var(--text-faint); }'
    '.cmp-title { font-family: var(--font-display); font-size: 26px; font-weight: 700;'
    ' color: var(--text-strong); margin: 6px 0 4px; }'
    '.cmp-title .cmp-vs { color: var(--orange-500); }'
    '.cmp-sub { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); }'
    '.cmp-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }'
    '.cmp-charts .panel { margin: 0; }'
    '.cmp-table { width: 100%; border-collapse: collapse; font-size: 13px; }'
    '.cmp-table th { font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;'
    ' letter-spacing: .06em; color: var(--text-faint); text-align: left; padding: 6px 10px;'
    ' border-bottom: 1px solid var(--border-subtle); }'
    '.cmp-table td { padding: 7px 10px; border-bottom: 1px solid var(--border-subtle); color: var(--text-body); }'
    '.cmp-diff-row:hover { background: var(--app-gray-100); }'
    '.cmp-delta { font-variant-numeric: tabular-nums; font-family: var(--font-mono); }'
    '.cmp-up { color: #0f9d6b; } .cmp-down { color: #df5325; } .cmp-flat { color: var(--text-faint); }'
    '.cmp-flip { color: #df5325; font-weight: 600; font-size: 11px; margin-left: 6px; font-family: var(--font-mono); }'
    '.cmp-unmatched { margin-top: 12px; font-size: 12px; color: var(--text-muted); }'
    '.cmp-unmatched ul { margin: 4px 0 0; padding-left: 18px; }'
    '.cmp-transcript-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }'
    '.cmp-side-title { font-family: var(--font-mono); font-size: 11px; text-transform: uppercase;'
    ' letter-spacing: .06em; color: var(--text-muted); margin: 0 0 8px; }'
    '</style>'
)


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def register_sim_compare_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register the two compare routes on *app* (called from ``build_app``)."""

    @app.get('/compare/sim')
    def compare_sim(req: Request) -> Response:
        rid_a = req.query_params.get('a') or ''
        rid_b = req.query_params.get('b') or ''
        if not rid_a or not rid_b:
            body = '<section class="cmp-header"><h1>Run comparison</h1><p class="sim-empty">Pick two runs to compare.</p></section>'
            return Response(page('Compare', body, active_surface='sim'), status_code=400, media_type='text/html')

        run_a = _load_run(rid_a, roots)
        run_b = _load_run(rid_b, roots)
        if run_a is None or run_b is None:
            missing = 'A' if run_a is None else 'B'
            body = f'<section class="cmp-header"><h1>Run comparison</h1><p class="sim-empty">Run {missing} not found or not a simulation run.</p></section>'
            return Response(page('Compare', body, active_surface='sim'), status_code=404, media_type='text/html')

        body = render_compare_page(rid_a, rid_b, run_a, run_b)
        return Response(page('Compare', body, active_surface='sim'), media_type='text/html')

    @app.get('/compare/sim/transcript')
    def compare_sim_transcript(req: Request) -> Response:
        rid_a = req.query_params.get('a') or ''
        rid_b = req.query_params.get('b') or ''
        try:
            ia = int(req.query_params.get('ia', ''))
            ib = int(req.query_params.get('ib', ''))
        except (TypeError, ValueError):
            return Response('<p class="sim-empty">Invalid conversation index.</p>', media_type='text/html')

        run_a = _load_run(rid_a, roots)
        run_b = _load_run(rid_b, roots)
        if run_a is None or run_b is None:
            return Response('<p class="sim-empty">Run not found.</p>', status_code=404, media_type='text/html')

        entries_a = _entries(run_a)
        entries_b = _entries(run_b)
        ea = next((e for e in entries_a if e.index == ia), None)
        eb = next((e for e in entries_b if e.index == ib), None)
        if ea is None or eb is None:
            return Response('<p class="sim-empty">No conversation at that index.</p>', media_type='text/html')

        return Response(render_compare_transcript(ea, eb, run_a.run_name, run_b.run_name), media_type='text/html')
