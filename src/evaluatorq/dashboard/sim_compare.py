"""Side-by-side comparison of two simulation runs (RES-1085).

The single-run sim views live in ``sim_views.py``; this module adds a compare
surface on top of the same ``sim`` adapter/library. Nothing here touches the sim
runner or data model — it only reads two already-stored ``SimulationRun`` objects.

Routes (registered by ``register_sim_compare_routes``):

    GET /compare/sim?a={ridA}&b={ridB}
        Full page: hero, KPI delta cards, A-vs-B charts, an all-metrics table,
        and an outcome-diff table matching conversations by ``(persona, scenario)``.

    GET /compare/sim/transcript?a={ridA}&b={ridB}&ia={idxA}&ib={idxB}
        Side-by-side transcript fragment for one matched conversation pair.

Conversations are matched by ``(persona, scenario)`` (the ticket's chosen key):
runs that share a dataset line up; anything unmatched is listed separately (with a
low-overlap warning) so a mismatched dataset is visible rather than silently dropped.

Charts are Vega-Lite specs rendered server-side to SVG via the shared
``common.reports.vega`` helpers — no client JS, no ``/static`` dependency.
"""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import kpi_cards, pct
from evaluatorq.common.reports.palette import COLORS
from evaluatorq.common.reports.vega import render_svg, vl_bar_h, vl_grouped_bar
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.sim_views import _entries_from_run, render_transcript_fragment
from evaluatorq.dashboard.view import _panel

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.simulation.types import SimulationEntry, SimulationRun

_UP = COLORS['success_400']  # improved / higher in B
_DOWN = COLORS['red_400']  # regressed / lower in B


# ---------------------------------------------------------------------------
# KPI deltas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KpiRow:
    """One comparison row: a label, the two run values, and their delta.

    ``kind`` drives formatting: ``'rate'`` (0..1 → %), ``'score'`` (2dp),
    ``'turns'`` (1dp), ``'count'`` (int). ``a`` / ``b`` may be ``None`` when a
    metric was not measured by that run (e.g. a scorer only one run used); such a
    row renders ``n/a`` and carries no delta arrow rather than a fabricated 0.0.
    """

    label: str
    a: float | None
    b: float | None
    kind: str

    @property
    def delta(self) -> float | None:
        if self.a is None or self.b is None:
            return None
        return self.b - self.a


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_kpis(
    entries_a: list[SimulationEntry], entries_b: list[SimulationEntry], run_a: SimulationRun, run_b: SimulationRun
) -> list[KpiRow]:
    """Build the KPI delta rows: aggregates, per-scorer averages, terminated-by.

    Aggregates are computed from the entry lists (so they track the conversations
    actually present); per-scorer values come from each run's ``scorer_averages``.
    A scorer present in only one run keeps ``None`` on the other side — it is not
    a real 0.0 and must not read as a regression.
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

    # Per-scorer averages — union of keys, but a missing scorer stays None (n/a),
    # never 0.0, so a run that never measured it shows no fabricated regression.
    sa_a = run_a.scorer_averages or {}
    sa_b = run_b.scorer_averages or {}
    rows.extend(
        KpiRow(f'Scorer: {key}', sa_a.get(key), sa_b.get(key), 'score') for key in sorted(set(sa_a) | set(sa_b))
    )

    # Terminated-by distribution — counts per reason, union of labels.
    tb_a = Counter(e.terminated_by for e in entries_a)
    tb_b = Counter(e.terminated_by for e in entries_b)
    rows.extend(
        KpiRow(f'Terminated: {label}', float(tb_a[label]), float(tb_b[label]), 'count')
        for label in sorted(set(tb_a) | set(tb_b))
    )

    return rows


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return 'n/a'
    if kind == 'rate':
        return f'{value * 100:.0f}%'
    if kind == 'score':
        return f'{value:.2f}'
    if kind == 'turns':
        return f'{value:.1f}'
    return f'{value:.0f}'


def _fmt_delta(row: KpiRow) -> str:
    d = row.delta
    if d is None:
        return '<span class="cmp-delta cmp-flat">—</span>'
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

    @property
    def match_rate(self) -> float:
        """Share of all conversations that paired up (0..1). 1.0 = identical key sets."""
        total = 2 * len(self.matched) + len(self.a_only) + len(self.b_only)
        return (2 * len(self.matched) / total) if total else 0.0


def match_entries(entries_a: list[SimulationEntry], entries_b: list[SimulationEntry]) -> Matching:
    """Pair conversations across runs by ``(persona, scenario)``.

    Duplicate keys within a run are paired positionally (first A with first B,
    etc.) — an approximation surfaced by the low-overlap note on the page; any
    leftovers on either side fall into ``a_only`` / ``b_only``.
    """
    by_key_b: dict[tuple[str, str], list[SimulationEntry]] = defaultdict(list)
    for e in entries_b:
        by_key_b[e.persona, e.scenario].append(e)

    matched: list[MatchedPair] = []
    a_only: list[SimulationEntry] = []
    consumed_b: set[int] = set()
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


# --- Charts: Vega-Lite specs rendered server-side to SVG (no JS / no /static) ---


def _chart_panel(title: str, sub: str, spec: dict[str, Any]) -> str:
    """Render a Vega-Lite spec to SVG and wrap it in a titled panel. Omits the
    panel entirely when the chart can't render (empty data / vl-convert absent)."""
    svg = render_svg(spec)
    return _panel(title, sub, svg) if svg else ''


def _outcomes_chart(
    entries_a: list[SimulationEntry], entries_b: list[SimulationEntry], name_a: str, name_b: str
) -> str:
    """A-vs-B grouped bars for the two headline 0..1 metrics."""
    rate_a, score_a, _ = _agg(entries_a)
    rate_b, score_b, _ = _agg(entries_b)
    spec = vl_grouped_bar(
        categories=['Goal-achieved rate', 'Mean goal score'],
        series=[(name_a, [rate_a, score_a]), (name_b, [rate_b, score_b])],
        x_title='0 to 1',
    )
    return _chart_panel('Outcomes', 'Goal rate and mean score · A vs B', spec)


def _scorer_chart(run_a: SimulationRun, run_b: SimulationRun, name_a: str, name_b: str) -> str:
    """A-vs-B grouped bars over the scorers BOTH runs measured. Scorers used by
    only one run are listed, not charted as a fake 0.0."""
    sa_a = run_a.scorer_averages or {}
    sa_b = run_b.scorer_averages or {}
    shared = sorted(set(sa_a) & set(sa_b))
    if not shared:
        return ''
    spec = vl_grouped_bar(
        categories=shared,
        series=[(name_a, [sa_a[k] for k in shared]), (name_b, [sa_b[k] for k in shared])],
        x_title='Average score',
    )
    svg = render_svg(spec)
    if not svg:
        return ''
    only = sorted((set(sa_a) | set(sa_b)) - set(shared))
    note = f'<p class="cmp-note">Not compared (measured by one run only): {esc(", ".join(only))}</p>' if only else ''
    return _panel('Scorer averages', 'Per-scorer mean · shared scorers only', svg + note)


def _terminated_chart(
    entries_a: list[SimulationEntry], entries_b: list[SimulationEntry], name_a: str, name_b: str
) -> str:
    """A-vs-B grouped bars for how conversations ended."""
    tb_a = Counter(e.terminated_by for e in entries_a)
    tb_b = Counter(e.terminated_by for e in entries_b)
    labels = sorted(set(tb_a) | set(tb_b))
    if not labels:
        return ''
    spec = vl_grouped_bar(
        categories=labels,
        series=[(name_a, [float(tb_a[x]) for x in labels]), (name_b, [float(tb_b[x]) for x in labels])],
        x_title='Conversations',
    )
    return _chart_panel('How conversations ended', 'Terminated-by counts · A vs B', spec)


def _delta_chart(matching: Matching) -> str:
    """Per-conversation goal-score change (B - A) over matched pairs, sorted;
    green bars improved in B, red regressed."""
    if not matching.matched:
        return ''
    rows = sorted(
        (
            (f'{pair.a.persona}/{pair.a.scenario}', pair.b.goal_completion_score - pair.a.goal_completion_score)
            for pair in matching.matched
        ),
        key=operator.itemgetter(1),
    )
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [_UP if v >= 0 else _DOWN for v in values]
    spec = vl_bar_h(
        labels=labels,
        values=values,
        color=_UP,
        colors=colors,
        x_title='B minus A (goal score)',
        value_labels=[f'{v:+.2f}' for v in values],
    )
    return _chart_panel(
        'Per-conversation score change',
        'B minus A on matched conversations · green improved, red regressed',
        spec,
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


def _hero(run_a: SimulationRun, run_b: SimulationRun, matching: Matching) -> str:
    """Editorial header + a match summary line (with a low-overlap warning)."""
    date_a = run_a.created_at.strftime('%Y-%m-%d %H:%M')
    date_b = run_b.created_at.strftime('%Y-%m-%d %H:%M')
    n_matched, n_a, n_b = len(matching.matched), len(matching.a_only), len(matching.b_only)
    summary = f'{n_matched} matched · {n_a} only in A · {n_b} only in B'
    # A low match rate means the diff/transcript compare covers only a slice — say so.
    warn = (
        f' <span class="cmp-warn">low overlap ({matching.match_rate:.0%}) — the two runs share few '
        "(persona, scenario) pairs, so most conversations can't be compared directly.</span>"
        if matching.match_rate < 0.5
        else ''
    )
    return (
        '<div class="report-head"><a class="report-back" href="/?surface=sim">&larr; Agent sim runs</a></div>'
        '<section class="cmp-hero">'
        '<div class="cmp-kicker">Agent sim // run comparison</div>'
        f'<h1 class="cmp-title">{esc(run_a.run_name)} <span class="cmp-vs">vs</span> {esc(run_b.run_name)}</h1>'
        f'<p class="cmp-sub">A: {esc(run_a.run_name)} · {date_a} &nbsp;&nbsp; B: {esc(run_b.run_name)} · {date_b}</p>'
        f'</section><p class="cmp-note">{summary}.{warn}</p>'
    )


def render_compare_page(rid_a: str, rid_b: str, run_a: SimulationRun, run_b: SimulationRun) -> str:
    """Full compare-page body: hero, KPI deltas + charts, all-metrics, diffs."""
    entries_a = _entries_from_run(run_a)
    entries_b = _entries_from_run(run_b)
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
        f'{_hero(run_a, run_b, matching)}'
        f'{kpi_section}{charts}'
        f'{_delta_chart(matching)}'
        f'{_all_metrics_table(kpis, name_a, name_b)}'
        f'{_diff_table(rid_a, rid_b, matching)}'
    )


# ---------------------------------------------------------------------------
# Loading (distinguishes not-found from load-failure, unlike sim_views._load_run)
# ---------------------------------------------------------------------------


def _resolve_run(rid: str, roots: list[Path] | None) -> tuple[str, Any | None]:
    """Load a sim run by report id.

    Returns ``('ok', run)``, ``('missing', None)`` when the id resolves to no
    file or a valid-but-non-sim report, or ``('error', None)`` when a matching
    sim file fails to load/parse — including a *syntactically* corrupt file, so it
    is reported as corrupt (422) rather than "not found" (404). The error case is
    logged with a stack so a corrupt report is debuggable.
    """
    import json

    from evaluatorq.dashboard import library
    from evaluatorq.dashboard.surfaces import ADAPTERS

    path = library.resolve(rid, roots)
    if path is None:
        return 'missing', None
    # Strict read: a corrupt/unreadable file must surface as an error, not be masked
    # as an unknown surface (which load_surface does for lenient directory scans).
    try:
        surface, _raw = library.load_surface_strict(path)
    except (json.JSONDecodeError, OSError):
        logger.opt(exception=True).warning('compare: corrupt/unreadable sim report {}', rid)
        return 'error', None
    if surface != 'sim':
        return 'missing', None
    adapter = ADAPTERS.get('sim')
    if adapter is None:
        return 'missing', None
    try:
        return 'ok', adapter.load(path)
    except Exception:
        logger.opt(exception=True).warning('compare: failed to load sim report {}', rid)
        return 'error', None


def _error_page(message: str, status: int) -> Response:
    body = f'<section class="cmp-hero"><h1 class="cmp-title">Run comparison</h1><p class="sim-empty">{esc(message)}</p></section>'
    return Response(page('Compare', body, active_surface='sim'), status_code=status, media_type='text/html')


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
            return _error_page('Pick two runs to compare.', 400)
        if rid_a == rid_b:
            return _error_page('Pick two different runs to compare.', 400)

        status_a, run_a = _resolve_run(rid_a, roots)
        status_b, run_b = _resolve_run(rid_b, roots)
        if 'error' in (status_a, status_b):
            which = 'A' if status_a == 'error' else 'B'
            return _error_page(f'Run {which} could not be loaded (the report may be corrupt).', 422)
        if run_a is None or run_b is None:
            which = 'A' if run_a is None else 'B'
            return _error_page(f'Run {which} was not found or is not a simulation run.', 404)

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
            return Response(
                '<p class="sim-empty">Invalid conversation index.</p>', status_code=400, media_type='text/html'
            )

        status_a, run_a = _resolve_run(rid_a, roots)
        status_b, run_b = _resolve_run(rid_b, roots)
        if 'error' in (status_a, status_b):
            return Response(
                '<p class="sim-empty">A run could not be loaded (report may be corrupt).</p>',
                status_code=422,
                media_type='text/html',
            )
        if run_a is None or run_b is None:
            return Response('<p class="sim-empty">Run not found.</p>', status_code=404, media_type='text/html')

        entries_a = _entries_from_run(run_a)
        entries_b = _entries_from_run(run_b)
        ea = next((e for e in entries_a if e.index == ia), None)
        eb = next((e for e in entries_b if e.index == ib), None)
        if ea is None or eb is None:
            return Response(
                '<p class="sim-empty">No conversation at that index.</p>', status_code=404, media_type='text/html'
            )

        return Response(render_compare_transcript(ea, eb, run_a.run_name, run_b.run_name), media_type='text/html')
