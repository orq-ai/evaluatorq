"""Hero, headlines, tiles, outcome and score panels, filter strip and page assembly."""

from __future__ import annotations

from collections import Counter
from itertools import cycle
from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import esc, kpi_cards
from evaluatorq.dashboard.filters import _sim_options_from_results
from evaluatorq.dashboard.report_kit import panel
from evaluatorq.dashboard.report_tabs import sim_entity_dialog
from evaluatorq.dashboard.sim_compare.fmt import _delta_pill, _fmt_num, _fmt_rate, _run_key
from evaluatorq.dashboard.sim_compare.model import _FLAT, Comparison, MatchedPair, ScoreRow, Side, build_comparison
from evaluatorq.dashboard.sim_compare.state import TableState, _compare_url, _qs
from evaluatorq.dashboard.sim_compare.table import render_conversations, resolve_view
from evaluatorq.dashboard.view import GOAL_OUTCOME_LABEL, _render_sim_filter_rail, filter_fragment
from evaluatorq.simulation.metrics import TURN_METRICS

if TYPE_CHECKING:
    from evaluatorq.simulation.contracts import SimulationRun

# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def _hero(cmp: Comparison) -> str:
    """Report hero (shared .report-hero classes) plus an Insights-style status
    line: one chip per run, the match count, and a low-overlap warning chip."""

    def run_chip(which: str, side: Side) -> str:
        date = side.run.created_at.strftime('%Y-%m-%d %H:%M')
        n = len(side.run.results)
        return (
            f'<span class="cmp-run">{_run_key(which)}'
            f'<span class="cmp-run-name">{esc(side.name)}</span>'
            f'<span class="cmp-run-meta">{date} · {n} conversations</span></span>'
        )

    full = cmp.full
    n_matched, n_a, n_b = len(full.matched), len(full.a_only), len(full.b_only)
    unmatched = f' · {n_a} only in A · {n_b} only in B' if n_a or n_b else ''
    warn = (
        f'<span class="cmp-chip cmp-chip--warn" title="The two runs share few (persona, scenario) pairs, so most '
        f'conversations can\'t be compared directly.">low overlap ({full.match_rate:.0%})</span>'
        if full.match_rate < 0.5
        else ''
    )
    return (
        '<header class="report-hero">'
        '<p class="report-hero-kicker">Agent Simulation · Run comparison</p>'
        f'<h2 class="report-hero-title">{esc(cmp.a.name)} <span class="cmp-vs">vs</span> {esc(cmp.b.name)}</h2>'
        '</header>'
        '<div class="cmp-status">'
        f'{run_chip("A", cmp.a)}{run_chip("B", cmp.b)}'
        f'<span class="cmp-chip">{n_matched} matched{unmatched}</span>{warn}'
        '</div>'
    )


def _view_attrs(cmp: Comparison, view: str) -> str:
    """HTMX attributes that load the conversation table at *view* (keeping the filter)."""
    return (
        f'hx-get="/compare/sim/conversations?{esc(_qs(cmp, cmp_view=view))}" hx-include="#filter-form" '
        'hx-target="#cmp-conversations" hx-swap="outerHTML show:#cmp-conversations:top"'
    )


# Scorers that restate the goal rate; the "biggest score move" headline skips them.
_GOAL_SCORERS = frozenset({'goal_achieved'})


def _headlines(cmp: Comparison, view: str, *, oob: bool = False) -> str:
    """Insights-style "what changed" rows: a big number, a one-line finding with
    its reason, and an action that narrows the conversation table to *view*.
    The row whose view is showing reads "Showing these" and clicking it again
    returns the table to its default view. The table route re-renders this
    block out of band so the pressed row follows the segment control."""
    changes = cmp.changes
    regressed = [p for p, c in changes if c == 'regressed']
    improved = [p for p, c in changes if c == 'improved']
    errored = [p for p, c in changes if c == 'error']
    rows: list[str] = []

    def row(number: str, finding: str, why: str, *, tone: str = '', target: str | None = None, go: str = '') -> str:
        tone_cls = f' cmp-head--{tone}' if tone else ''
        on = target is not None and target == view
        if on:
            go_html = '<span class="cmp-head-go">Showing these &#x2713;</span>'
        elif go:
            go_html = f'<span class="cmp-head-go">{esc(go)} &rarr;</span>'
        else:
            go_html = '<span></span>'
        inner = (
            f'<span class="cmp-head-number">{esc(number)}</span>'
            f'<span><span class="cmp-head-finding">{esc(finding)}</span>'
            f'<span class="cmp-head-why">{esc(why)}</span></span>{go_html}'
        )
        if target is None:
            return f'<div class="cmp-head{tone_cls}">{inner}</div>'
        # Pressing the showing row again goes back to the default view ('' = changed, or all).
        attrs = _view_attrs(cmp, '' if on else target)
        return (
            f'<button type="button" class="cmp-head{tone_cls}{" is-on" if on else ""}" '
            f'aria-pressed="{"true" if on else "false"}" {attrs}>{inner}</button>'
        )

    def where(pairs: list[MatchedPair]) -> str:
        scenario, count = Counter(p.a.scenario for p in pairs).most_common(1)[0]
        return f' · most in {scenario} ({count} of {len(pairs)})' if count > 1 and len(pairs) > 1 else ''

    def plural(n: int) -> str:
        return f'{n} conversation{"" if n == 1 else "s"}'

    if regressed:
        rows.append(
            row(
                plural(len(regressed)),
                'Regressed in B',
                'Goal met in A, missed in B' + where(regressed),
                tone='bad',
                target='regressed',
                go='Review regressions',
            )
        )
    if improved:
        rows.append(
            row(
                plural(len(improved)),
                'Improved in B',
                'Goal missed in A, met in B' + where(improved),
                tone='good',
                target='improved',
                go='Review improvements',
            )
        )
    if errored:
        rows.append(
            row(
                plural(len(errored)),
                'Error changes',
                'A conversation errored in one run and not the other' + where(errored),
                tone='warn',
                target='error',
                go='Review errors',
            )
        )
    agg_a, agg_b = cmp.aggregates
    if cmp.outcome_filtered:
        rows.append(
            row(
                'n/a',
                'Goal rate not comparable',
                'A filter picks conversations by their goal outcome, so the rate measures the filter, not the runs',
                target='all',
                go='All conversations',
            )
        )
    elif agg_a.goal_rate is not None and agg_b.goal_rate is not None:
        pts = (agg_b.goal_rate - agg_a.goal_rate) * 100
        verb = 'held steady' if abs(pts) < 0.5 else ('rose' if pts > 0 else 'fell')
        rows.append(
            row(
                f'{pts:+.0f} pts',
                f'Goal rate {verb}',
                f'{agg_a.goal_rate:.0%} in A, {agg_b.goal_rate:.0%} in B',
                target='all',
                go='All conversations',
            )
        )
    # Only 0-1 rows: a turn count or token mean would outrank every scorer by scale alone.
    moved = [
        r
        for r in cmp.scores
        if r.in_unit_range and r.label not in _GOAL_SCORERS and r.delta is not None and abs(r.delta) >= _FLAT
    ]
    if moved:
        top = max(moved, key=lambda r: abs(r.delta or 0.0))
        delta = top.delta or 0.0
        better = (delta > 0) == top.higher_better
        rows.append(
            row(
                f'{delta:+.2f}',
                f'Biggest score move: {top.label} {"improved" if better else "worsened"}',
                f'{_fmt_num(top.a)} in A, {_fmt_num(top.b)} in B',
            )
        )
    if not regressed and not improved and not errored and cmp.matching.matched:
        rows.insert(0, row('0', 'No outcome changes', 'Every matched conversation kept its goal outcome'))
    oob_attr = ' hx-swap-oob="true"' if oob else ''
    return f'<section id="cmp-heads" class="cmp-heads" aria-label="What changed"{oob_attr}>{"".join(rows)}</section>'


_FILTER_LABELS = {
    'persona': 'Persona',
    'scenario': 'Scenario',
    'terminated_by': 'Ended by',
    'goal_outcome': 'Goal outcome',
    'rule_broken': 'Rule broken',
    'max_goal_score': 'Goal score',
    'min_turns': 'Turns',
    'min_total_tokens': 'Tokens',
}


def _filter_label(dim: str) -> str:
    if dim in _FILTER_LABELS:
        return _FILTER_LABELS[dim]
    metric = next((m for m in TURN_METRICS if dim in (f'min_{m.key}', f'max_{m.key}')), None)
    return metric.label if metric else dim.replace('_', ' ').capitalize()


def _filter_value_text(dim: str, values: list[str]) -> str:
    if dim == 'goal_outcome':
        return ', '.join(GOAL_OUTCOME_LABEL.get(v, v) for v in values)
    if dim.startswith('max_'):
        return f'≤ {values[0]}'
    if dim.startswith('min_'):
        return f'≥ {values[0]}'
    if dim == 'rule_broken':
        return 'yes'
    shown = ', '.join(values[:2])
    return shown + (f' +{len(values) - 2}' if len(values) > 2 else '')


def _filter_strip(cmp: Comparison, state: TableState, *, oob: bool = False) -> str:
    """Insights-style filter bar: how many rows the filters keep, one removable
    chip per active filter, clear all, and a copy-link button (the URL holds
    the filters, segment and sort). Chips swap the page like the rail does and
    stay plain links without JavaScript; the table route re-renders this bar
    out of band so the links carry the current segment and sort."""
    full = cmp.full
    noun = 'rows' if full.a_only or full.b_only else 'pairs'
    count = f'<span class="cmp-fbar-count">Showing <b>{cmp.shown_rows}</b> of <b>{cmp.total_rows}</b> {noun}</span>'
    base = TableState(state.view, state.sort, state.direction, 1, state.size)
    if cmp.narrowing:
        chips = ''.join(
            f'<span class="cmp-fchip"><span class="cmp-fchip-dim">{esc(_filter_label(dim))}</span>'
            f'{esc(_filter_value_text(dim, values))}'
            f'<a class="cmp-fchip-x" {_swap_link(cmp, {k: v for k, v in cmp.narrowing.items() if k != dim}, base)} '
            f'aria-label="Remove the {esc(_filter_label(dim))} filter">&times;</a></span>'
            for dim, values in cmp.narrowing.items()
        )
        rest = f'{chips}<a class="cmp-fbar-clear" {_swap_link(cmp, {}, base)}>Clear all</a>'
    else:
        rest = '<span class="cmp-fbar-hint">Use the filters to narrow both runs to the same conversations.</span>'
    copy = '<button type="button" class="cmp-fbar-copy" data-copy-link>Copy link to this view</button>'
    oob_attr = ' hx-swap-oob="true"' if oob else ''
    return f'<div id="cmp-fbar" class="cmp-fbar" role="status"{oob_attr}>{count}{rest}{copy}</div>'


def _swap_link(cmp: Comparison, narrowing: dict[str, list[str]], state: TableState) -> str:
    """``href`` + HTMX attributes that load the compare view for *narrowing*
    through the filter route, swapping the page and rail like a rail change."""
    url = _compare_url(cmp, narrowing, state)
    query = url.partition('?')[2]
    return f'href="{esc(url)}" hx-get="/compare/sim/filter?{esc(query)}" hx-target="#filter-swap" hx-swap="outerHTML"'


def _tiles(cmp: Comparison) -> str:
    """Single-run style number tiles: B's value large, then ``A → B`` and a delta
    pill. Under an outcome filter the goal tiles say "not comparable" instead."""
    a, b = cmp.aggregates

    def tile(
        label: str,
        va: float | None,
        vb: float | None,
        fmt: Any,
        delta_text: Any,
        *,
        higher: bool | None,
        outcome: bool = False,
    ) -> dict:
        d = None if va is None or vb is None else vb - va
        if outcome and cmp.outcome_filtered:
            pill = '<span class="cmp-delta cmp-delta--flat" title="A filter picks conversations by goal outcome">not comparable</span>'
        else:
            pill = _delta_pill(d, delta_text(d) if d is not None else '', higher_better=higher)
        return {
            'label': label,
            'value': fmt(vb),
            'sub_html': f'<span class="cmp-tile-ab">{esc(fmt(va))} &rarr; {esc(fmt(vb))}</span>{pill}',
        }

    return kpi_cards(
        [
            tile(
                'Conversations',
                a.conversations,
                b.conversations,
                lambda v: f'{v:.0f}',
                lambda d: f'{d:+.0f}',
                higher=None,
            ),
            tile(
                'Goal rate',
                a.goal_rate,
                b.goal_rate,
                _fmt_rate,
                lambda d: f'{d * 100:+.0f} pts',
                higher=True,
                outcome=True,
            ),
            tile(
                'Avg goal score',
                a.avg_score,
                b.avg_score,
                _fmt_num,
                lambda d: f'{d:+.2f}',
                higher=True,
                outcome=True,
            ),
            tile('Avg turns', a.avg_turns, b.avg_turns, lambda v: _fmt_num(v, 1), lambda d: f'{d:+.1f}', higher=None),
            tile('Errors', a.errors, b.errors, lambda v: f'{v:.0f}', lambda d: f'{d:+.0f}', higher=False),
        ],
        modifier='cmp',
    )


def _stack_rows(cmp: Comparison, segments_for: Any) -> str:
    """Two stacked 100% bars (A, B) plus a legend that carries each category's
    ``A → B`` count — the Insights "at a glance" bar. Counts live in the legend,
    not on the segments, where small text over the fills fails contrast."""
    segs_a: list[tuple[str, str, int]] = segments_for(cmp.a)
    segs_b: list[tuple[str, str, int]] = segments_for(cmp.b)

    def bar(which: str, counts: list[tuple[str, str, int]]) -> str:
        total = sum(n for _, _, n in counts)
        if not total:
            body = '<span class="cmp-stack-empty">No conversations</span>'
        else:
            body = ''.join(
                f'<i style="flex-grow:{n};background:{color}" title="{esc(label)}: {n} of {total}"></i>'
                for label, color, n in counts
                if n
            )
        return (
            f'<div class="cmp-stack-row"><span class="cmp-stack-key">{_run_key(which)}</span>'
            f'<div class="cmp-stack-bar">{body}</div><span class="cmp-stack-total">{total}</span></div>'
        )

    legend_html = ''.join(
        f'<span><i style="background:{color}"></i>{esc(label)}<b class="cmp-legend-n">{na} &rarr; {nb}</b></span>'
        for (label, color, na), (_, _, nb) in zip(segs_a, segs_b, strict=True)
        if na or nb
    )
    return (
        f'<div class="cmp-stack">{bar("A", segs_a)}{bar("B", segs_b)}</div><div class="cmp-legend">{legend_html}</div>'
    )


_OUTCOME_SEGMENTS = (
    ('met', 'Goal met', 'var(--outcome-resistant)'),
    ('missed', 'Goal missed', 'var(--outcome-vulnerable)'),
    ('error', 'Error', 'var(--outcome-error)'),
)
# Termination reasons are categories, not verdicts — neutral inks, except the
# error reason, which keeps the error amber.
_TERMINATION_INKS = ('var(--ink-900)', 'var(--text-muted)', 'var(--teal-600)', 'var(--text-body)')


def _outcomes_panel(cmp: Comparison) -> str:
    def segments(side: Side) -> list[tuple[str, str, int]]:
        counts = Counter(side.status(e) for e in side.entries)
        return [(label, color, counts[key]) for key, label, color in _OUTCOME_SEGMENTS]

    reasons = sorted({e.terminated_by for s in (cmp.a, cmp.b) for e in s.entries})
    colors = {}
    inks = cycle(_TERMINATION_INKS)
    for reason in reasons:
        colors[reason] = 'var(--outcome-error)' if reason == 'error' else next(inks)

    def term_segments(side: Side) -> list[tuple[str, str, int]]:
        counts = Counter(e.terminated_by for e in side.entries)
        return [(reason, colors[reason], counts[reason]) for reason in reasons]

    outcomes = _stack_rows(cmp, segments)
    # One termination reason is a 100% bar that says nothing; leave it out.
    ended = _stack_rows(cmp, term_segments) if len(reasons) > 1 else ''
    body = f'<div class="cmp-subhead">Goal outcome</div>{outcomes}' + (
        f'<div class="cmp-subhead">How conversations ended</div>{ended}' if ended else ''
    )
    return panel('Outcomes', body, sub='Filtered conversations per run')


def _scores_panel(rows: list[ScoreRow]) -> str:
    """Scorer and turn-quality means: A and B as numbers, on a shared 0..1 track
    where the row fits it, plus the delta."""
    if not rows:
        return panel('Scores', '<p class="sim-empty">Neither run recorded scorer or turn-quality data.</p>')

    def failed(v: float | None, dropped: int) -> bool:
        return v is None and dropped > 0

    def value_text(v: float | None, dropped: int) -> str:
        return 'failed' if failed(v, dropped) else _fmt_num(v)

    def drop_title(which: str, dropped: int) -> str:
        return f' title="{which}: scorer failed on {dropped} conversation{"" if dropped == 1 else "s"}"'

    def track(v: float | None, which: str, dropped: int) -> str:
        if failed(v, dropped):
            return f'<div class="cmp-track"{drop_title(which, dropped)}><span class="cmp-track-na">failed</span></div>'
        if v is None:
            return '<div class="cmp-track"><span class="cmp-track-na">n/a</span></div>'
        return (
            f'<div class="cmp-track" title="{which}: {v:.2f}">'
            f'<i class="cmp-track-fill cmp-track-fill--{which.lower()}" style="width:{v * 100:.1f}%"></i></div>'
        )

    def ab_text(r: ScoreRow) -> str:
        parts = []
        for which, v, dropped in (('A', r.a, r.dropped_a), ('B', r.b, r.dropped_b)):
            text = value_text(v, dropped)
            parts.append(
                f'<span class="cmp-score-failed"{drop_title(which, dropped)}>{text}</span>'
                if failed(v, dropped)
                else text
            )
        return f'{parts[0]} &rarr; {parts[1]}'

    body = ''.join(
        '<div class="cmp-score-row">'
        f'<div class="cmp-score-label">{esc(r.label)}'
        + ('' if r.higher_better else '<span class="cmp-score-hint">lower is better</span>')
        + '</div>'
        + (
            f'<div class="cmp-score-tracks">{track(r.a, "A", r.dropped_a)}{track(r.b, "B", r.dropped_b)}</div>'
            if r.in_unit_range
            else '<div class="cmp-score-tracks"></div>'
        )
        + f'<div class="cmp-score-ab">{ab_text(r)}</div>'
        f'<div>{_delta_pill(r.delta, f"{r.delta:+.2f}" if r.delta is not None else "", higher_better=r.higher_better)}</div>'
        '</div>'
        for r in rows
    )
    one_sided = [
        r.label for r in rows if r.delta is None and not failed(r.a, r.dropped_a) and not failed(r.b, r.dropped_b)
    ]
    note = f'<p class="cmp-note">Measured by one run only: {esc(", ".join(one_sided))}</p>' if one_sided else ''
    legend = f'<div class="cmp-legend"><span>{_run_key("A")}</span><span>{_run_key("B")}</span></div>'
    sub = 'Mean per conversation, 0 to 1' if all(r.in_unit_range for r in rows) else 'Mean per conversation'
    return panel('Scores', f'{legend}<div class="cmp-scores">{body}</div>{note}', sub=sub)


def render_compare_body(cmp: Comparison, state: TableState) -> str:
    """Everything left of the filter rail."""
    return (
        f'{_hero(cmp)}'
        f'{_filter_strip(cmp, state)}'
        f'{_headlines(cmp, resolve_view(cmp, state))}'
        f'{_tiles(cmp)}'
        f'<div class="cmp-grid">{_outcomes_panel(cmp)}{_scores_panel(cmp.scores)}</div>'
        f'{render_conversations(cmp, state)}'
    )


def render_filter_rail(cmp: Comparison, selections: dict[str, list[str]]) -> str:
    """The sim filter rail, posting to the compare filter route; options cover both runs."""
    opts = _sim_options_from_results([*cmp.a.run.results, *cmp.b.run.results])
    m = cmp.matching
    return _render_sim_filter_rail(
        cmp.a.rid,
        opts,
        selections,
        shown=len(m.matched) + len(m.a_only) + len(m.b_only),
        total=cmp.total_rows,
        post_url=f'/compare/sim/filter?{_qs(cmp)}',
        include='#cmp-table-state',
        counter_note="A pair stays when either run's conversation matches.",
    )


def _swap(cmp: Comparison, selections: dict[str, list[str]], state: TableState) -> str:
    return filter_fragment(cmp.a.rid, 'sim', render_compare_body(cmp, state), render_filter_rail(cmp, selections))


def render_compare_page(
    rid_a: str,
    rid_b: str,
    run_a: SimulationRun,
    run_b: SimulationRun,
    selections: dict[str, list[str]] | None = None,
    state: TableState | None = None,
) -> str:
    """Full compare-page body: the filter swap container plus the drawer dialog.

    The dialog sits outside ``#filter-swap`` so a filter change does not tear
    down an open drawer; both live inside ``.sim-report`` for the drawer styles.
    """
    selections = selections or {}
    cmp = build_comparison(rid_a, rid_b, run_a, run_b, selections)
    return (
        '<div class="report-aligned sim-report cmp-report">'
        f'<section class="report-view">{_swap(cmp, selections, state or TableState())}</section>'
        f'{sim_entity_dialog("sim-entity-dialog--wide")}'
        '</div>'
    )
