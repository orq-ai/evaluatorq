"""Side-by-side comparison of two simulation runs.

The single-run sim views live in ``sim_views.py``; this module adds a compare
surface on top of the same ``sim`` adapter/library. Nothing here touches the sim
runner or data model — it only reads two already-stored ``SimulationRun`` objects.

Routes (registered by ``register_sim_compare_routes``):

    GET  /compare/sim?a={ridA}&b={ridB}[&<filter params>][&cmp_view=&cmp_sort=…]
        Full page: hero with a run/match status line, "what changed" headline
        rows, number tiles with A → B deltas, outcome and termination bars, a
        scores panel, the conversation table, and the sim filter rail.

    POST /compare/sim/filter?a={ridA}&b={ridB}
        Filter-rail swap: re-renders the body and the rail from the filtered
        conversations of both runs. Sets ``HX-Push-Url`` to the canonical GET
        URL of the resulting view, so the address bar is shareable.

    GET  /compare/sim/conversations?a={ridA}&b={ridB}&cmp_view=&cmp_sort=…
        The conversation table fragment (segment, sort, page, size), with the
        current filter state pulled in via ``hx-include``. Also sets ``HX-Push-Url``.

    GET  /compare/sim/transcript?a={ridA}&b={ridB}&ia={idxA}&ib={idxB}
        Paired transcript for one matched conversation, loaded into the drawer.

Conversations are matched by ``(persona, scenario)``: runs that share a dataset
line up; anything unmatched is listed separately (with a low-overlap warning) so
a mismatched dataset is visible rather than silently dropped.

The filter decides which rows survive: a matched pair stays when either of its
conversations passes, an unmatched conversation stays when it passes. Every
number on the page is computed over each run's side of the surviving rows, so
both runs are always described by the same conversations the table lists.

Everything renders as server-side HTML and CSS — no chart library.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.matching import pair_indices_by_key
from evaluatorq.common.reports import esc, kpi_cards, status_badge
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.sim_views import (
    _entries_from_run,
    _sim_header_row,
    _sim_pager,
    filtered_entries,
    render_transcript_fragment,
)
from evaluatorq.dashboard.view import report_back_link
from evaluatorq.simulation.metrics import TURN_METRICS, is_errored_result

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.simulation.contracts import SimulationEntry, SimulationResult, SimulationRun

_EPS = 1e-9


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
    matched_indices, a_only_indices, b_only_indices = pair_indices_by_key(
        entries_a,
        entries_b,
        key=lambda entry: (entry.persona, entry.scenario),
    )
    return Matching(
        [MatchedPair(entries_a[i], entries_b[j]) for i, j in matched_indices],
        [entries_a[i] for i in a_only_indices],
        [entries_b[i] for i in b_only_indices],
    )


# ---------------------------------------------------------------------------
# Comparison model: both runs, filtered, plus the matching over them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Side:
    """One run's view: the entries that survived the filter (pair-level, see
    ``build_comparison``) and which original indices errored (``is_errored_result``
    needs the raw result)."""

    rid: str
    run: SimulationRun
    entries: list[SimulationEntry]
    errored: frozenset[int]
    results: list[SimulationResult]
    """The raw results behind ``entries``, built once for the sections builders."""

    @property
    def name(self) -> str:
        return self.run.run_name

    def status(self, entry: SimulationEntry) -> str:
        """``'error'`` / ``'met'`` / ``'missed'`` — the sim row-list's outcome vocabulary."""
        if entry.index in self.errored:
            return 'error'
        return 'met' if entry.goal_achieved else 'missed'


@dataclass(frozen=True)
class Comparison:
    a: Side
    b: Side
    matching: Matching
    """Surviving rows: a pair stays when either side passes the filter, an unmatched entry when it passes."""
    full: Matching
    """The unfiltered matching, for the status line and the rail's shown/total counter."""

    @property
    def total_rows(self) -> int:
        """Rows (pairs + unmatched) before filtering."""
        return len(self.full.matched) + len(self.full.a_only) + len(self.full.b_only)


def _side(rid: str, run: SimulationRun, entries: list[SimulationEntry]) -> Side:
    errored = frozenset(i for i, r in enumerate(run.results) if is_errored_result(result=r))
    entries = sorted(entries, key=lambda e: e.index)
    return Side(rid, run, entries, errored, [run.results[e.index] for e in entries])


def build_comparison(
    rid_a: str, rid_b: str, run_a: SimulationRun, run_b: SimulationRun, selections: dict[str, list[str]]
) -> Comparison:
    full = match_entries(_entries_from_run(run_a), _entries_from_run(run_b))
    keep_a = {e.index for e in filtered_entries(run_a, selections)}
    keep_b = {e.index for e in filtered_entries(run_b, selections)}
    matching = Matching(
        [p for p in full.matched if p.a.index in keep_a or p.b.index in keep_b],
        [e for e in full.a_only if e.index in keep_a],
        [e for e in full.b_only if e.index in keep_b],
    )
    side_a = _side(rid_a, run_a, [*(p.a for p in matching.matched), *matching.a_only])
    side_b = _side(rid_b, run_b, [*(p.b for p in matching.matched), *matching.b_only])
    return Comparison(side_a, side_b, matching, full)


def pair_change(a: Side, b: Side, pair: MatchedPair) -> str:
    """``'regressed'`` (met -> missed), ``'improved'`` (missed -> met), ``'error'``
    (the statuses differ and either side errored), or ``'same'``."""
    status_a, status_b = a.status(pair.a), b.status(pair.b)
    if status_a == status_b:
        return 'same'
    if 'error' in (status_a, status_b):
        return 'error'
    return 'regressed' if status_a == 'met' else 'improved'


@dataclass(frozen=True)
class Aggregates:
    conversations: int
    goal_rate: float | None
    avg_score: float | None
    avg_turns: float | None
    errors: int


def aggregates(side: Side) -> Aggregates:
    """Headline numbers for one run's surviving conversations, taken from the
    single-run summary section so they match the run's own report by
    construction: errored conversations count in the denominator of the goal
    rate and the average score. An empty side has no rates (n/a)."""
    from evaluatorq.simulation.reports.sections import _build_summary_section

    if not side.results:
        return Aggregates(0, None, None, None, 0)
    data = _build_summary_section(side.results).data
    return Aggregates(
        conversations=data['total_conversations'],
        goal_rate=data['success_rate'],
        avg_score=data['avg_goal_completion_score'],
        avg_turns=data['avg_turn_count'],
        errors=data['errors'],
    )


@dataclass(frozen=True)
class ScoreRow:
    """One scores-panel row. ``higher_better`` is False for risk metrics.
    ``dropped_*`` counts conversations whose scorer failed in that run, so a
    ``None`` mean with drops reads as failed rather than not measured."""

    label: str
    a: float | None
    b: float | None
    higher_better: bool = True
    dropped_a: int = 0
    dropped_b: int = 0

    @property
    def delta(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a

    @property
    def in_unit_range(self) -> bool:
        return all(v is None or 0.0 <= v <= 1.0 for v in (self.a, self.b))


def _scorer_stats(side: Side) -> dict[str, tuple[float | None, int]]:
    """Scorer name -> (mean score, dropped count) over the side's results."""
    from evaluatorq.simulation.reports.sections import _build_evaluator_scores_section

    section = _build_evaluator_scores_section(side.results)
    if section is None:
        return {}
    return {row['evaluator']: (row['mean_score'], row['dropped']) for row in section.data['rows']}


def _quality_means(side: Side) -> dict[str, float]:
    from evaluatorq.simulation.reports.sections import _build_turn_metrics_section

    return dict(_build_turn_metrics_section(side.results).data['avg_quality_metrics'])


def score_rows(cmp: Comparison) -> list[ScoreRow]:
    """Per-scorer and per-turn-quality means over each run's surviving
    conversations. A metric one run never measured stays ``None`` there (n/a),
    never 0.0, so it cannot read as a regression; a scorer that failed on every
    conversation is ``None`` with a drop count."""
    sa, sb = _scorer_stats(cmp.a), _scorer_stats(cmp.b)
    rows = []
    for key in sorted(set(sa) | set(sb)):
        mean_a, dropped_a = sa.get(key, (None, 0))
        mean_b, dropped_b = sb.get(key, (None, 0))
        rows.append(ScoreRow(key, mean_a, mean_b, dropped_a=dropped_a, dropped_b=dropped_b))
    qa, qb = _quality_means(cmp.a), _quality_means(cmp.b)
    rows.extend(
        ScoreRow(m.label, qa.get(m.key), qb.get(m.key), higher_better=not m.high_is_risky)
        for m in TURN_METRICS
        if m.key in qa or m.key in qb
    )
    return rows


# ---------------------------------------------------------------------------
# Small renderers
# ---------------------------------------------------------------------------


def _delta_pill(delta: float | None, text: str, *, higher_better: bool | None = True) -> str:
    """Signed change pill. Colour follows good/bad (green/red), not direction;
    ``higher_better=None`` keeps it neutral (e.g. conversation count, turns)."""
    if delta is None:
        return '<span class="cmp-delta cmp-delta--flat">n/a</span>'
    if abs(delta) < _EPS:
        return '<span class="cmp-delta cmp-delta--flat">no change</span>'
    arrow = '&#x25B2;' if delta > 0 else '&#x25BC;'
    tone = 'flat' if higher_better is None else 'good' if (delta > 0) == higher_better else 'bad'
    return f'<span class="cmp-delta cmp-delta--{tone}">{arrow} {esc(text)}</span>'


def _run_key(which: str) -> str:
    """The A/B identity marker: a grey (A, baseline) or ink (B) square + letter."""
    return f'<span class="cmp-key cmp-key--{which.lower()}" aria-hidden="true"></span><b class="cmp-key-letter">{which}</b>'


_STATUS_BADGE = {
    'met': ('Goal met', 'pass'),
    'missed': ('Goal missed', 'fail'),
    'error': ('Error', 'warn'),
}


def _badge(status: str) -> str:
    text, tone = _STATUS_BADGE[status]
    return status_badge(text, tone)


_CHANGE_PILL = {
    'regressed': ('Regressed', 'bad'),
    'improved': ('Improved', 'good'),
    'error': ('Error', 'warn'),
    'same': ('Same', 'flat'),
}


def _change_pill(change: str) -> str:
    text, tone = _CHANGE_PILL[change]
    return f'<span class="cmp-change cmp-change--{tone}">{text}</span>'


def _fmt_rate(v: float | None) -> str:
    return 'n/a' if v is None else f'{v:.0%}'


def _fmt_num(v: float | None, digits: int = 2) -> str:
    return 'n/a' if v is None else f'{v:.{digits}f}'


def _qs(cmp: Comparison, **params: Any) -> str:
    return urlencode({'a': cmp.a.rid, 'b': cmp.b.rid, **params})


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


def _headlines(cmp: Comparison, scores: list[ScoreRow]) -> str:
    """Insights-style "what changed" rows: a big number, a one-line finding with
    its reason, and an action that narrows the conversation table."""
    changes = [(p, pair_change(cmp.a, cmp.b, p)) for p in cmp.matching.matched]
    regressed = [p for p, c in changes if c == 'regressed']
    improved = [p for p, c in changes if c == 'improved']
    errored = [p for p, c in changes if c == 'error']
    rows: list[str] = []

    def row(number: str, finding: str, why: str, *, tone: str = '', view: str | None = None, go: str = '') -> str:
        tone_cls = f' cmp-head--{tone}' if tone else ''
        go_html = f'<span class="cmp-head-go">{esc(go)} &rarr;</span>' if go else '<span></span>'
        inner = (
            f'<span class="cmp-head-number">{esc(number)}</span>'
            f'<span><span class="cmp-head-finding">{esc(finding)}</span>'
            f'<span class="cmp-head-why">{esc(why)}</span></span>{go_html}'
        )
        if view is None:
            return f'<div class="cmp-head{tone_cls}">{inner}</div>'
        return f'<button type="button" class="cmp-head{tone_cls}" {_view_attrs(cmp, view)}>{inner}</button>'

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
                view='regressed',
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
                view='improved',
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
                view='error',
                go='Review errors',
            )
        )
    agg_a, agg_b = aggregates(cmp.a), aggregates(cmp.b)
    if agg_a.goal_rate is not None and agg_b.goal_rate is not None:
        pts = (agg_b.goal_rate - agg_a.goal_rate) * 100
        verb = 'held steady' if abs(pts) < 0.5 else ('rose' if pts > 0 else 'fell')
        rows.append(
            row(
                f'{pts:+.0f} pts',
                f'Goal rate {verb}',
                f'{agg_a.goal_rate:.0%} in A, {agg_b.goal_rate:.0%} in B',
                view='all',
                go='All conversations',
            )
        )
    moved = [r for r in scores if r.delta is not None and abs(r.delta) >= 0.01]
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
    if not rows:
        return ''
    return f'<section class="cmp-heads" aria-label="What changed">{"".join(rows)}</section>'


def _tiles(cmp: Comparison) -> str:
    """Single-run style number tiles: B's value large, then ``A → B`` and a delta pill."""
    a, b = aggregates(cmp.a), aggregates(cmp.b)

    def tile(label: str, va: float | None, vb: float | None, fmt: Any, delta_text: Any, *, higher: bool | None) -> dict:
        d = None if va is None or vb is None else vb - va
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
            tile('Goal rate', a.goal_rate, b.goal_rate, _fmt_rate, lambda d: f'{d * 100:+.0f} pts', higher=True),
            tile('Avg score', a.avg_score, b.avg_score, _fmt_num, lambda d: f'{d:+.2f}', higher=True),
            tile('Avg turns', a.avg_turns, b.avg_turns, lambda v: _fmt_num(v, 1), lambda d: f'{d:+.1f}', higher=None),
            tile('Errors', a.errors, b.errors, lambda v: f'{v:.0f}', lambda d: f'{d:+.0f}', higher=False),
        ],
        modifier='cmp',
    )


def _stack_rows(cmp: Comparison, segments_for: Any, legend: list[tuple[str, str]]) -> str:
    """Two stacked 100% bars (A, B) plus a dot legend — the Insights "at a glance" bar."""

    def bar(which: str, side: Side) -> str:
        counts: list[tuple[str, str, int]] = segments_for(side)
        total = sum(n for _, _, n in counts)
        if not total:
            body = '<span class="cmp-stack-empty">No conversations</span>'
        else:
            body = ''.join(
                f'<i style="flex-grow:{n};background:{color}" title="{esc(label)}: {n}">'
                f'{n if n / total >= 0.07 else ""}</i>'
                for label, color, n in counts
                if n
            )
        return (
            f'<div class="cmp-stack-row"><span class="cmp-stack-key">{_run_key(which)}</span>'
            f'<div class="cmp-stack-bar">{body}</div><span class="cmp-stack-total">{total}</span></div>'
        )

    legend_html = ''.join(f'<span><i style="background:{color}"></i>{esc(label)}</span>' for label, color in legend)
    return f'<div class="cmp-stack">{bar("A", cmp.a)}{bar("B", cmp.b)}</div><div class="cmp-legend">{legend_html}</div>'


_OUTCOME_SEGMENTS = (
    ('met', 'Goal met', 'var(--outcome-resistant)'),
    ('missed', 'Goal missed', 'var(--outcome-vulnerable)'),
    ('error', 'Error', 'var(--outcome-error)'),
)
# Termination reasons are categories, not verdicts — neutral inks, except the
# error reason, which keeps the error amber. All are dark enough for the white count labels.
_TERMINATION_INKS = ('var(--ink-900)', 'var(--text-muted)', 'var(--teal-600)', 'var(--text-body)')


def _outcomes_panel(cmp: Comparison) -> str:
    from evaluatorq.dashboard.report_kit import panel

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

    outcomes = _stack_rows(cmp, segments, [(label, color) for _, label, color in _OUTCOME_SEGMENTS])
    ended = _stack_rows(cmp, term_segments, [(r, colors[r]) for r in reasons]) if reasons else ''
    body = f'<div class="cmp-subhead">Goal outcome</div>{outcomes}' + (
        f'<div class="cmp-subhead">How conversations ended</div>{ended}' if ended else ''
    )
    return panel('Outcomes', body, sub='Filtered conversations per run')


def _scores_panel(rows: list[ScoreRow]) -> str:
    """Scorer and turn-quality means: A and B as numbers, on a shared 0..1 track
    where the row fits it, plus the delta."""
    from evaluatorq.dashboard.report_kit import panel

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


# ---------------------------------------------------------------------------
# Conversation table
# ---------------------------------------------------------------------------

_VIEWS = (
    ('changed', 'Changed'),
    ('regressed', 'Regressed'),
    ('improved', 'Improved'),
    ('error', 'Errors'),
    ('all', 'All'),
)
_VIEW_KEYS = {key for key, _ in _VIEWS}

# Worst first: a regression outranks an error, which outranks an improvement.
_SEVERITY = {'regressed': 0, 'error': 1, 'improved': 2, 'same': 3}

# (param, header label, sort key over (pair, change))
_COLUMNS: list[tuple[str, str, Any]] = [
    ('persona', 'Persona', lambda pc: pc[0].a.persona.lower()),
    ('scenario', 'Scenario', lambda pc: pc[0].a.scenario.lower()),
    ('score', 'Score A → B', lambda pc: pc[0].b.goal_completion_score - pc[0].a.goal_completion_score),
    ('turns', 'Turns A → B', lambda pc: pc[0].b.turn_count - pc[0].a.turn_count),
    (
        'change',
        'Change',
        lambda pc: (_SEVERITY[pc[1]], pc[0].b.goal_completion_score - pc[0].a.goal_completion_score),
    ),
]
_SORT_KEYS = {param: key for param, _, key in _COLUMNS}
_DEFAULT_SORT = 'change'  # regressions first, then by the biggest score drop
_PAGE_SIZES = (10, 25, 50)
_DEFAULT_SIZE = 10


@dataclass(frozen=True)
class TableState:
    view: str = ''
    sort: str = _DEFAULT_SORT
    direction: str = 'asc'
    page: int = 1
    size: int = _DEFAULT_SIZE

    @classmethod
    def parse(cls, params: Any) -> TableState:
        """Clamp request params (query string or form) to valid values."""
        view = params.get('cmp_view') or ''
        sort = params.get('cmp_sort') or _DEFAULT_SORT
        direction = params.get('cmp_dir') or 'asc'
        try:
            page = int(params.get('cmp_page') or 1)
        except (TypeError, ValueError):
            page = 1
        try:
            size = int(params.get('cmp_size') or _DEFAULT_SIZE)
        except (TypeError, ValueError):
            size = _DEFAULT_SIZE
        return cls(
            view if view in _VIEW_KEYS else '',
            sort if sort in _SORT_KEYS else _DEFAULT_SORT,
            direction if direction in ('asc', 'desc') else 'asc',
            max(1, page),
            size if size in _PAGE_SIZES else _DEFAULT_SIZE,
        )


def _in_view(change: str, view: str) -> bool:
    if view == 'all':
        return True
    if view == 'changed':
        return change != 'same'
    return change == view


def render_conversations(cmp: Comparison, state: TableState) -> str:
    """The conversation section: segment control, sortable/paged table whose
    rows open the paired transcript in the drawer, and the unmatched lists."""
    changes = [(p, pair_change(cmp.a, cmp.b, p)) for p in cmp.matching.matched]
    counts = {key: sum(1 for _, c in changes if _in_view(c, key)) for key, _ in _VIEWS}
    # Default to the changed conversations — the reason to open a comparison —
    # unless nothing changed, in which case show everything rather than nothing.
    view = state.view or ('changed' if counts['changed'] else 'all')

    def control(sort: str, direction: str, page: int, size: int, *, v: str = view) -> str:
        qs = _qs(cmp, cmp_view=v, cmp_sort=sort, cmp_dir=direction, cmp_page=page, cmp_size=size)
        return (
            f'hx-get="/compare/sim/conversations?{esc(qs)}" hx-include="#filter-form" '
            'hx-target="#cmp-conversations" hx-swap="outerHTML"'
        )

    seg = ''.join(
        (
            f'<button type="button" class="cmp-seg-btn" aria-pressed="true" disabled>{label}'
            f'<span class="cmp-seg-n">{counts[key]}</span></button>'
            if key == view
            else f'<button type="button" class="cmp-seg-btn" aria-pressed="false" '
            f'{control(state.sort, state.direction, 1, state.size, v=key)}>{label}'
            f'<span class="cmp-seg-n">{counts[key]}</span></button>'
        )
        for key, label in _VIEWS
    )

    visible = sorted(
        ((p, c) for p, c in changes if _in_view(c, view)),
        key=_SORT_KEYS[state.sort],
        reverse=state.direction == 'desc',
    )
    total = len(visible)
    pages = max(1, -(-total // state.size))
    page_no = min(state.page, pages)
    start = (page_no - 1) * state.size

    rows: list[str] = []
    for pair, change in visible[start : start + state.size]:
        a, b = pair.a, pair.b
        d_score = b.goal_completion_score - a.goal_completion_score
        d_turns = b.turn_count - a.turn_count
        url = f'/compare/sim/transcript?{_qs(cmp, ia=a.index, ib=b.index)}'
        rows.append(
            f'<tr class="sim-conv-row cmp-conv-row cmp-conv-row--{change}" role="button" tabindex="0" '
            f'data-sim-entity-trigger data-entity-kind="conversation" data-drawer-url="{esc(url)}">'
            f'<td class="sim-conv-persona">{esc(a.persona)}</td>'
            f'<td class="sim-conv-scenario">{esc(a.scenario)}</td>'
            f'<td class="cmp-num">{a.goal_completion_score:.2f} &rarr; {b.goal_completion_score:.2f}'
            f' <span class="cmp-num-d">{d_score:+.2f}</span></td>'
            f'<td class="cmp-num">{a.turn_count} &rarr; {b.turn_count} <span class="cmp-num-d">{d_turns:+d}</span></td>'
            f'<td>{_change_pill(change)}</td>'
            f'<td class="cmp-outcome">{_badge(cmp.a.status(a))}<span class="cmp-arrow">&rarr;</span>{_badge(cmp.b.status(b))}</td>'
            '</tr>'
        )

    if rows:
        header = _sim_header_row(
            control, state.sort, state.direction, state.size, columns=_COLUMNS, trailing=('Outcome A → B',)
        )
        table = (
            '<div class="sim-conv-table-shell">'
            f'<table class="sim-conv-table cmp-conv-table">{header}<tbody>{"".join(rows)}</tbody></table>'
            '</div>'
            f'{_sim_pager(control, state.sort, state.direction, page_no, pages, total, state.size, noun="pairs", sizes=_PAGE_SIZES)}'
        )
    elif not cmp.matching.matched:
        table = '<p class="sim-empty">No conversations matched on (persona, scenario).</p>'
    else:
        table = f'<p class="sim-empty">No {view} conversations under the current filters.</p>'

    def unmatched(title: str, which: str, entries: list[SimulationEntry]) -> str:
        if not entries:
            return ''
        items = ''.join(f'<li>{esc(e.persona)} &middot; {esc(e.scenario)}</li>' for e in entries)
        return (
            f'<details class="cmp-unmatched"><summary>{_run_key(which)} {esc(title)} '
            f'<span class="cmp-seg-n">{len(entries)}</span></summary><ul>{items}</ul></details>'
        )

    # The filter form hx-includes this block, so a filter change keeps the
    # current segment, sort and page size (the page resets to 1).
    state_inputs = ''.join(
        f'<input type="hidden" name="{name}" value="{esc(str(value))}">'
        for name, value in (
            ('cmp_view', view),
            ('cmp_sort', state.sort),
            ('cmp_dir', state.direction),
            ('cmp_size', state.size),
        )
    )
    return (
        '<section id="cmp-conversations" class="rk-panel cmp-conversations">'
        '<div class="cmp-conv-head"><div>'
        '<div class="rk-panel-title">Conversations</div>'
        '<div class="rk-panel-sub">Matched by persona and scenario · click a row to compare transcripts</div>'
        f'</div><div class="cmp-seg" role="group" aria-label="Show">{seg}</div></div>'
        f'<div id="cmp-table-state" hidden>{state_inputs}</div>'
        f'{table}'
        f'{unmatched("Only in A", "A", cmp.matching.a_only)}{unmatched("Only in B", "B", cmp.matching.b_only)}'
        '</section>'
    )


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def render_compare_body(cmp: Comparison, state: TableState) -> str:
    """Everything left of the filter rail."""
    scores = score_rows(cmp)
    return (
        f'{_hero(cmp)}'
        f'{_headlines(cmp, scores)}'
        f'{_tiles(cmp)}'
        f'<div class="cmp-grid">{_outcomes_panel(cmp)}{_scores_panel(scores)}</div>'
        f'{render_conversations(cmp, state)}'
    )


def render_filter_rail(cmp: Comparison, selections: dict[str, list[str]]) -> str:
    """The sim filter rail, posting to the compare filter route; options cover both runs."""
    from evaluatorq.dashboard.filters import _sim_options_from_results
    from evaluatorq.dashboard.view import _render_sim_filter_rail

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
        counter_note='A pair shows when either run matches.',
    )


def _swap(cmp: Comparison, selections: dict[str, list[str]], state: TableState) -> str:
    from evaluatorq.dashboard.view import filter_fragment

    return filter_fragment(cmp.a.rid, 'sim', render_compare_body(cmp, state), render_filter_rail(cmp, selections))


def canonical_url(cmp: Comparison, selections: dict[str, list[str]], state: TableState) -> str:
    """The GET ``/compare/sim`` URL that reproduces this slice and table state.

    Selections use the repeated ``dimension=value`` form ``parse_selections``
    reads, limited to filters that drop a conversation; table params equal to
    their defaults are left out.
    """
    from evaluatorq.dashboard.filters import FILTERS

    sim = FILTERS['sim']

    def narrows(dim: str) -> bool:
        # The rail posts every control, defaults included; keep only the ones that drop a conversation.
        picked = {dim: selections[dim]}
        return any(len(sim.apply(run, picked)) < len(run.results) for run in (cmp.a.run, cmp.b.run))

    params: list[tuple[str, str]] = [('a', cmp.a.rid), ('b', cmp.b.rid)]
    params.extend(
        (dim, value) for dim in sim.dimensions if selections.get(dim) and narrows(dim) for value in selections[dim]
    )
    table = (
        ('cmp_view', state.view, ''),
        ('cmp_sort', state.sort, _DEFAULT_SORT),
        ('cmp_dir', state.direction, 'asc'),
        ('cmp_page', state.page, 1),
        ('cmp_size', state.size, _DEFAULT_SIZE),
    )
    params.extend((name, str(value)) for name, value, default in table if value != default)
    return f'/compare/sim?{urlencode(params)}'


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
    from evaluatorq.dashboard.report_tabs import sim_entity_dialog

    selections = selections or {}
    cmp = build_comparison(rid_a, rid_b, run_a, run_b, selections)
    return (
        '<div class="report-aligned sim-report cmp-report">'
        f'<section class="report-view">{_swap(cmp, selections, state or TableState())}</section>'
        f'{sim_entity_dialog("sim-entity-dialog--wide")}'
        '</div>'
    )


def render_compare_transcript(entry_a: SimulationEntry, entry_b: SimulationEntry, side_a: Side, side_b: Side) -> str:
    """Paired transcript for the drawer: a header naming the pair and its
    change, then the two single-run transcript fragments side by side."""

    def column(which: str, entry: SimulationEntry, side: Side) -> str:
        return (
            '<div class="cmp-side">'
            f'<div class="cmp-side-title">{_run_key(which)}<span class="cmp-side-name">{esc(side.name)}</span>'
            f'{_badge(side.status(entry))}</div>'
            f'{render_transcript_fragment(entry)}</div>'
        )

    change = pair_change(side_a, side_b, MatchedPair(entry_a, entry_b))
    head = (
        '<header class="cmp-pair-head">'
        f'<div class="cmp-pair-title">{esc(entry_a.persona)} <span class="cmp-vs">·</span> {esc(entry_a.scenario)}</div>'
        f'{_change_pill(change)}</header>'
    )
    return f'{head}<div class="cmp-transcript-grid">{column("A", entry_a, side_a)}{column("B", entry_b, side_b)}</div>'


# ---------------------------------------------------------------------------
# Loading (distinguishes not-found from load-failure, unlike sim_views._load_run)
# ---------------------------------------------------------------------------


def resolve_run(rid: str, roots: list[Path] | None, surface: str) -> tuple[str, Any | None]:
    """Load a run of the given *surface* ('sim' or 'pairwise') by report id.

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
        found, _raw = library.load_surface_strict(path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        logger.opt(exception=True).warning('compare: corrupt/unreadable {} report {}', surface, rid)
        return 'error', None
    if found != surface:
        return 'missing', None
    adapter = ADAPTERS.get(surface)
    if adapter is None:
        return 'missing', None
    try:
        return 'ok', adapter.load(path)
    except Exception:
        logger.opt(exception=True).warning('compare: failed to load {} report {}', surface, rid)
        return 'error', None


def error_page(message: str, status: int, surface: str) -> Response:
    """Full-page compare error for *surface* ('sim' or 'pairwise')."""
    body = (
        '<header class="report-hero"><h2 class="report-hero-title">Run comparison</h2></header>'
        f'<p class="sim-empty">{esc(message)}</p>'
    )
    return Response(
        page('Compare', body, active_surface=surface, back_html=report_back_link(surface)),
        status_code=status,
        media_type='text/html',
    )


def _fragment_error(message: str, status: int) -> Response:
    return Response(f'<p class="sim-empty">{esc(message)}</p>', status_code=status, media_type='text/html')


def _load_pair(
    rid_a: str, rid_b: str, roots: list[Path] | None
) -> tuple[SimulationRun, SimulationRun] | tuple[str, int]:
    """Both runs, or ``(message, status)`` naming which one failed and why."""
    if not rid_a or not rid_b:
        return 'Pick two runs to compare.', 400
    if rid_a == rid_b:
        return 'Pick two different runs to compare.', 400
    status_a, run_a = resolve_run(rid_a, roots, 'sim')
    status_b, run_b = resolve_run(rid_b, roots, 'sim')
    if 'error' in (status_a, status_b):
        which = 'A' if status_a == 'error' else 'B'
        return f'Run {which} could not be loaded (the report may be corrupt).', 422
    if run_a is None or run_b is None:
        which = 'A' if run_a is None else 'B'
        return f'Run {which} was not found or is not a simulation run.', 404
    return run_a, run_b


def _push(response: Response, url: str) -> Response:
    """Tell HTMX to put *url* in the address bar for this swap."""
    response.headers['HX-Push-Url'] = url
    return response


def _filter_response(roots: list[Path] | None, rid_a: str, rid_b: str, items: list[tuple[str, str]]) -> Response:
    """Blocking half of the filter route (disk loads and rendering), run off the event loop."""
    loaded = _load_pair(rid_a, rid_b, roots)
    if isinstance(loaded[0], str):
        message, status = loaded
        return _fragment_error(str(message), int(status))
    run_a, run_b = loaded
    selections: dict[str, list[str]] = {}
    table: dict[str, str] = {}
    for key, value in items:
        if key.startswith('cmp_'):
            table[key] = value
        else:
            selections.setdefault(key, []).append(value)
    state = TableState.parse(table)
    cmp = build_comparison(rid_a, rid_b, run_a, run_b, selections)
    return _push(Response(_swap(cmp, selections, state), media_type='text/html'), canonical_url(cmp, selections, state))


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def register_sim_compare_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register the compare routes on *app* (called from ``build_app``)."""
    from evaluatorq.dashboard.filter_request import parse_selections
    from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope

    def _ids(req: Request) -> tuple[str, str]:
        return req.query_params.get('a') or '', req.query_params.get('b') or ''

    @app.get('/compare/sim')
    def compare_sim(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        loaded = _load_pair(rid_a, rid_b, roots)
        if isinstance(loaded[0], str):
            message, status = loaded
            return error_page(str(message), int(status), 'sim')
        run_a, run_b = loaded
        body = render_compare_page(
            rid_a, rid_b, run_a, run_b, parse_selections(req, 'sim'), TableState.parse(req.query_params)
        )
        return Response(
            page('Compare', body, active_surface='sim', back_html=report_back_link('sim')), media_type='text/html'
        )

    @app.post('/compare/sim/filter')
    async def compare_sim_filter(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        form = await req.form()
        items = [(key, str(value)) for key, value in form.multi_items()]
        return await asyncio.to_thread(_filter_response, roots, rid_a, rid_b, items)

    @app.get('/compare/sim/conversations')
    def compare_sim_conversations(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        loaded = _load_pair(rid_a, rid_b, roots)
        if isinstance(loaded[0], str):
            message, status = loaded
            return _fragment_error(str(message), int(status))
        run_a, run_b = loaded
        selections = parse_selections(req, 'sim')
        state = TableState.parse(req.query_params)
        cmp = build_comparison(rid_a, rid_b, run_a, run_b, selections)
        return _push(
            Response(render_conversations(cmp, state), media_type='text/html'), canonical_url(cmp, selections, state)
        )

    @app.get('/compare/sim/transcript')
    def compare_sim_transcript(req: Request) -> Response:
        rid_a, rid_b = _ids(req)
        try:
            ia = int(req.query_params.get('ia', ''))
            ib = int(req.query_params.get('ib', ''))
        except (TypeError, ValueError):
            return _fragment_error('Invalid conversation index.', 400)
        loaded = _load_pair(rid_a, rid_b, roots)
        if isinstance(loaded[0], str):
            message, status = loaded
            return _fragment_error(str(message), int(status))
        run_a, run_b = loaded
        # Row links carry original indices, so resolve against the full, unfiltered runs.
        full = match_entries(_entries_from_run(run_a), _entries_from_run(run_b))
        pair = next((p for p in full.matched if p.a.index == ia and p.b.index == ib), None)
        if pair is None:
            return _fragment_error('Not a matched pair.', 404)
        side_a = _side(rid_a, run_a, [pair.a])
        side_b = _side(rid_b, run_b, [pair.b])
        with cli_slug_render_scope():
            body = render_compare_transcript(pair.a, pair.b, side_a, side_b)
        return Response(body, media_type='text/html')
