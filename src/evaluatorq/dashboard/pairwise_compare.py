"""Side-by-side comparison of two pairwise runs.

Route (registered by ``register_pairwise_compare_routes``):

    GET /compare/pairwise?a={ridA}&b={ridB}

Comparisons are matched across runs by the exact ``question`` string; the
status codes mirror ``/compare/sim``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.html_helpers import kpi_cards, pct
from evaluatorq.dashboard.pairwise_palette import pw_side_label, verdict_eyebrow
from evaluatorq.dashboard.report_kit import panel
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.sim_compare import error_page, resolve_run
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
    total: int

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


def _headline(run: PairwiseRun) -> str:
    v = verdict(run)
    eyebrow = verdict_eyebrow(v.state)
    if v.leader_label is None:
        return eyebrow
    return f'{eyebrow}: {v.leader_label}'


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
    return panel(f'{side} · {run.run_name}', body, sub=f'{run.label_a} vs {run.label_b}')


def _link(rid: str, n: int) -> str:
    return f'<a href="/r/{esc(rid)}?cmp={n}#cmp-{n}">#{n}</a>'


def _diff_table(rid_a: str, rid_b: str, run_a: PairwiseRun, run_b: PairwiseRun, m: PairwiseMatching) -> str:
    if not m.matched:
        return panel('Matched comparisons', '<p class="pw-empty">No question appears in both runs.</p>')
    rows: list[str] = []
    for i, j in m.matched:
        ea, eb = run_a.entries[i], run_b.entries[j]
        wa = pw_side_label(ea.comparison.winner, run_a)
        wb = pw_side_label(eb.comparison.winner, run_b)
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


# ---------------------------------------------------------------------------
# Loading and routes
# ---------------------------------------------------------------------------


def register_pairwise_compare_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register ``GET /compare/pairwise`` on *app* (called from ``build_app``)."""

    @app.get('/compare/pairwise')
    def compare_pairwise(req: Request) -> Response:
        rid_a = req.query_params.get('a') or ''
        rid_b = req.query_params.get('b') or ''
        if not rid_a or not rid_b:
            return error_page('Pick two runs to compare.', 400, 'pairwise')
        if rid_a == rid_b:
            return error_page('Pick two different runs to compare.', 400, 'pairwise')

        status_a, run_a = resolve_run(rid_a, roots, 'pairwise')
        status_b, run_b = resolve_run(rid_b, roots, 'pairwise')
        if 'error' in (status_a, status_b):
            which = 'A' if status_a == 'error' else 'B'
            return error_page(f'Run {which} could not be loaded (the report may be corrupt).', 422, 'pairwise')
        if run_a is None or run_b is None:
            which = 'A' if run_a is None else 'B'
            return error_page(f'Run {which} was not found or is not a pairwise run.', 404, 'pairwise')

        body = render_pairwise_compare_page(rid_a, rid_b, run_a, run_b)
        return Response(
            page('Compare', body, active_surface='pairwise', back_html=report_back_link('pairwise')),
            media_type='text/html',
        )
