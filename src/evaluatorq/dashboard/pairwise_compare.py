"""Side-by-side comparison of two pairwise runs.

Route (registered by ``register_pairwise_compare_routes``):

    GET /compare/pairwise?a={ridA}&b={ridB}

Comparisons are matched across runs by the exact ``question`` string; the
status codes mirror ``/compare/sim``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import kpi_cards, pct
from evaluatorq.dashboard.report_kit import panel
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.view import report_back_link
from evaluatorq.pairwise_reports.stats import fleiss_kappa, kappa_label, verdict

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.pairwise_run import PairwiseRun

_MATCH_WARN_BELOW = 0.5


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseMatching:
    matched: list[tuple[int, int]]
    a_only: list[int]
    b_only: list[int]
    total: int = 0

    @property
    def match_rate(self) -> float:
        return len(self.matched) / max(self.total, 1)


def match_comparisons(a: PairwiseRun, b: PairwiseRun) -> PairwiseMatching:
    """Pair entries by exact ``question``; duplicates pair positionally per key."""
    b_by_q: dict[str, list[int]] = defaultdict(list)
    for j, e in enumerate(b.entries):
        b_by_q[e.question].append(j)
    cursor: dict[str, int] = defaultdict(int)
    matched: list[tuple[int, int]] = []
    a_only: list[int] = []
    used_b: set[int] = set()
    for i, e in enumerate(a.entries):
        pool = b_by_q.get(e.question, [])
        k = cursor[e.question]
        if k < len(pool):
            matched.append((i, pool[k]))
            used_b.add(pool[k])
            cursor[e.question] = k + 1
        else:
            a_only.append(i)
    b_only = [j for j in range(len(b.entries)) if j not in used_b]
    total = max(len(a.entries), len(b.entries), 1)
    return PairwiseMatching(matched, a_only, b_only, total)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _winner_label(run: PairwiseRun, winner: str) -> str:
    if winner == 'A':
        return run.label_a
    if winner == 'B':
        return run.label_b
    return winner


def _headline(run: PairwiseRun) -> str:
    v = verdict(run)
    if v.state == 'no_decided':
        return 'No decided comparisons'
    if v.leader_label is None:
        return 'Even'
    prefix = '✓ Clear winner' if v.state == 'clear' else '⚠ Too close to call'
    return f'{prefix}: {v.leader_label}'


def _run_panel(side: str, run: PairwiseRun) -> str:
    v = verdict(run)
    total = len(run.entries)
    share = pct(v.decided / total) if total else '—'
    if v.decided:
        rates = f'{esc(run.label_a)} {pct(v.wins_a / v.decided)} · {esc(run.label_b)} {pct(v.wins_b / v.decided)}'
    else:
        rates = '—'
    kappa, rows = fleiss_kappa(run)
    kappa_txt = '—' if kappa is None or rows == 0 else f'{kappa:.2f} ({esc(kappa_label(kappa))})'
    body = (
        f'<p class="pw-cmp-verdict">{esc(_headline(run))}</p>'
        '<dl class="pw-cmp-facts">'
        f'<dt>Win rate</dt><dd>{rates}</dd>'
        f'<dt>Decided</dt><dd>{share}</dd>'
        f'<dt>Judge agreement (κ)</dt><dd>{kappa_txt}</dd>'
        '</dl>'
    )
    return panel(f'{side} · {run.run_name}', body, sub=f'{esc(run.label_a)} vs {esc(run.label_b)}')


def _link(rid: str, n: int) -> str:
    return f'<a href="/r/{esc(rid)}#cmp-{n}">#{n}</a>'


def _diff_table(rid_a: str, rid_b: str, run_a: PairwiseRun, run_b: PairwiseRun, m: PairwiseMatching) -> str:
    if not m.matched:
        return panel('Matched comparisons', '<p class="pw-empty">No question appears in both runs.</p>')
    rows: list[str] = []
    for i, j in m.matched:
        ea, eb = run_a.entries[i], run_b.entries[j]
        wa = _winner_label(run_a, ea.comparison.winner)
        wb = _winner_label(run_b, eb.comparison.winner)
        changed = '<span class="tag pw-changed">changed</span>' if wa != wb else ''
        rows.append(
            f'<tr><td>{_link(rid_a, i + 1)}</td><td>{_link(rid_b, j + 1)}</td>'
            f'<td class="pw-cmp-q">{esc(ea.question)}</td>'
            f'<td>{esc(wa)}</td><td>{esc(wb)} {changed}</td></tr>'
        )
    table = (
        '<table class="pw-cmp-table"><thead><tr>'
        '<th>#A</th><th>#B</th><th>Question</th><th>Winner in run A</th><th>Winner in run B</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )
    return panel('Matched comparisons', table, sub=f'{len(m.matched)} compared')


def render_pairwise_compare_page(rid_a: str, rid_b: str, run_a: PairwiseRun, run_b: PairwiseRun) -> str:
    """Compare-page body for two pairwise runs."""
    m = match_comparisons(run_a, run_b)
    date_a = run_a.created_at.strftime('%Y-%m-%d %H:%M')
    date_b = run_b.created_at.strftime('%Y-%m-%d %H:%M')
    warn = (
        f' <span class="cmp-warn">low overlap ({m.match_rate:.0%}) — the two runs share few questions, '
        "so most comparisons can't be set against each other.</span>"
        if m.match_rate < _MATCH_WARN_BELOW
        else ''
    )
    hero = (
        '<header class="report-hero">'
        '<p class="report-hero-kicker">Pairwise · Run comparison</p>'
        f'<h2 class="report-hero-title">{esc(run_a.run_name)} <span class="cmp-vs">vs</span> {esc(run_b.run_name)}</h2>'
        f'<p class="report-hero-sub">A: {esc(run_a.run_name)} · {date_a} &nbsp;&nbsp; B: {esc(run_b.run_name)} · {date_b}</p>'
        '</header>'
        f'<p class="cmp-note">{len(m.matched)} of {m.total} comparisons matched by question.{warn}</p>'
    )
    columns = f'<div class="pw-cmp-cols">{_run_panel("A", run_a)}{_run_panel("B", run_b)}</div>'
    leftovers = kpi_cards([
        {'label': 'Only in run A', 'value': str(len(m.a_only)), 'status': 'neutral'},
        {'label': 'Only in run B', 'value': str(len(m.b_only)), 'status': 'neutral'},
    ])
    return (
        f'<div class="report-aligned pw-report pw-compare">{hero}{columns}'
        f'{_diff_table(rid_a, rid_b, run_a, run_b, m)}{leftovers}</div>'
    )


def pairwise_run_compare_control(rid: str, choices: list[tuple[str, str]]) -> str:
    """On-report compare control: this run is A, pick B from the other runs."""
    others = [(r, name) for r, name in choices if r != rid]
    if not others:
        return ''
    opts = ''.join(f'<option value="{esc(r)}">{esc(name)}</option>' for r, name in others)
    return (
        '<form class="cmp-bar" action="/compare/pairwise" method="get">'
        f'<input type="hidden" name="a" value="{esc(rid)}">'
        '<span class="cmp-bar-label">Compare with</span>'
        f'<select name="b" aria-label="Run B">{opts}</select>'
        '<button type="submit" class="btn-secondary">Compare</button>'
        '</form>'
    )


# ---------------------------------------------------------------------------
# Loading and routes
# ---------------------------------------------------------------------------


def _resolve_run(rid: str, roots: list[Path] | None) -> tuple[str, Any | None]:
    """Load a pairwise run: ``('ok', run)``, ``('missing', None)`` or ``('error', None)``."""
    from evaluatorq.dashboard import library
    from evaluatorq.dashboard.surfaces import ADAPTERS

    path = library.resolve(rid, roots)
    if path is None:
        return 'missing', None
    try:
        surface, _raw = library.load_surface_strict(path)
    except (json.JSONDecodeError, OSError):
        logger.opt(exception=True).warning('compare: corrupt/unreadable pairwise report {}', rid)
        return 'error', None
    if surface != 'pairwise':
        return 'missing', None
    adapter = ADAPTERS.get('pairwise')
    if adapter is None:
        return 'missing', None
    try:
        return 'ok', adapter.load(path)
    except Exception:  # noqa: BLE001 — any loader failure is a 422, logged with stack
        logger.opt(exception=True).warning('compare: failed to load pairwise report {}', rid)
        return 'error', None


def _error_page(message: str, status: int) -> Response:
    body = (
        '<header class="report-hero"><h2 class="report-hero-title">Run comparison</h2></header>'
        f'<p class="sim-empty">{esc(message)}</p>'
    )
    return Response(
        page('Compare', body, active_surface='pairwise', back_html=report_back_link('pairwise')),
        status_code=status,
        media_type='text/html',
    )


def register_pairwise_compare_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register ``GET /compare/pairwise`` on *app* (called from ``build_app``)."""

    @app.get('/compare/pairwise')
    def compare_pairwise(req: Request) -> Response:
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
            return _error_page(f'Run {which} was not found or is not a pairwise run.', 404)

        body = render_pairwise_compare_page(rid_a, rid_b, run_a, run_b)
        return Response(
            page('Compare', body, active_surface='pairwise', back_html=report_back_link('pairwise')),
            media_type='text/html',
        )
