"""Pairwise comparisons list: dot strip, paged row list and lazy detail rows.

Routes (HTML fragments, no page shell):

    GET /r/{rid}/pairwise/row-list   → filtered, sorted, paged row list
    GET /r/{rid}/pairwise/comparison → one comparison's detail (full-run index)
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any

from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.filter_request import parse_selections
from evaluatorq.dashboard.filters import apply_or_all
from evaluatorq.dashboard.pairwise_palette import pw_side_colour, pw_side_label
from evaluatorq.dashboard.report_kit import panel
from evaluatorq.dashboard.trace_links import single_trace_url, trace_link_button
from evaluatorq.pairwise_reports.export_html import _classify_details
from evaluatorq.pairwise_reports.sections import _classify_observations, judge_order
from evaluatorq.pairwise_reports.stats import comparison_flags

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.pairwise import PairwiseVote
    from evaluatorq.pairwise_run import PairwiseEntry, PairwiseRun

_PAGE_SIZE = 25
_SORTS = ('index', 'winner')
_WINNER_ORDER = {'A': 0, 'B': 1, 'tie': 2, 'inconclusive': 3}


def _short(model: str) -> str:
    return model.split('/')[-1]


# ---------------------------------------------------------------------------
# Row list
# ---------------------------------------------------------------------------


def _chips(entry: PairwiseEntry, run: PairwiseRun, judges: list[str]) -> str:
    by_model = {v.model: v for v in entry.comparison.votes}
    chips = []
    for judge in judges:
        vote = by_model.get(judge)
        value = vote.vote if vote else None
        cls = 'pw-vchip is-none' if value is None else 'pw-vchip'
        chips.append(
            f'<span class="{cls}" style="--c:{pw_side_colour(value)}" title="{esc(judge)}">'
            f'{esc(_short(judge))} → {esc(pw_side_label(value, run))}</span>'
        )
    return ''.join(chips)


def _hx(rid: str, sort: str, direction: str, page: int) -> str:
    return (
        f'hx-get="/r/{esc(rid)}/pairwise/row-list?sort={sort}&amp;dir={direction}&amp;page={page}"'
        f' hx-include="#filter-form" hx-target="#pw-row-list-{esc(rid)}" hx-swap="outerHTML"'
    )


def _sort_button(rid: str, label: str, key: str, sort: str, direction: str) -> str:
    active = sort == key
    next_dir = 'desc' if active and direction == 'asc' else 'asc'
    arrow = (' ▴' if direction == 'asc' else ' ▾') if active else ''
    aria = ('ascending' if direction == 'asc' else 'descending') if active else 'none'
    return (
        f'<button type="button" class="pw-th-sort" aria-sort="{aria}" {_hx(rid, key, next_dir, 1)}>'
        f'{label}{arrow}</button>'
    )


def _pager(rid: str, sort: str, direction: str, page: int, pages: int, total: int) -> str:
    if pages <= 1:
        return f'<nav class="pw-pager"><span>{total} comparisons</span></nav>'
    prev_attrs = 'disabled' if page <= 1 else _hx(rid, sort, direction, page - 1)
    next_attrs = 'disabled' if page >= pages else _hx(rid, sort, direction, page + 1)
    return (
        '<nav class="pw-pager">'
        f'<button type="button" class="pw-pager-btn" {prev_attrs}>‹ Prev</button>'  # noqa: RUF001
        f'<span>Page {page} of {pages} · {total} comparisons</span>'
        f'<button type="button" class="pw-pager-btn" {next_attrs}>Next ›</button>'  # noqa: RUF001
        '</nav>'
    )


def _row(rid: str, run: PairwiseRun, judges: list[str], n: int, entry: PairwiseEntry, *, with_trace: bool) -> str:
    winner = entry.comparison.winner
    trace = ''
    if with_trace:
        trace = trace_link_button(
            single_trace_url(entry.comparison.trace_id),
            'Trace ↗',
            onclick='event.stopPropagation()',
        )
    summary = (
        '<summary class="pw-cmp-summary">'
        f'<span class="pw-cmp-n">{n}</span>'
        f'<span class="pw-cmp-q">{esc(entry.question)}</span>'
        f'<span class="pw-cmp-win"><span class="pw-vchip" style="--c:{pw_side_colour(winner)}">'
        f'{esc(pw_side_label(winner, run))}</span></span>'
        f'<span class="pw-chiprow">{_chips(entry, run, judges)}</span>'
        f'<span class="pw-cmp-trace">{trace}</span>'
        '</summary>'
    )
    cols = 5 if with_trace else 4
    return (
        f'<tbody class="pw-cmp-group" id="cmp-{n}"><tr><td colspan="{cols}">'
        '<details class="pw-cmp-row"'
        f' hx-get="/r/{esc(rid)}/pairwise/comparison?idx={n - 1}"'
        ' hx-trigger="toggle once" hx-target="find .pw-cmp-detail" hx-swap="innerHTML">'
        f'{summary}<div class="pw-cmp-detail"></div></details></td></tr></tbody>'
    )


def render_pairwise_row_list(
    rid: str,
    run: PairwiseRun,
    entries: list[PairwiseEntry],
    *,
    sort: str = 'index',
    direction: str = 'asc',
    page: int = 1,
    page_size: int = _PAGE_SIZE,
) -> str:
    """Render the paged row list for ``entries`` (a subset of ``run.entries``)."""
    sort = sort if sort in _SORTS else 'index'
    direction = direction if direction in ('asc', 'desc') else 'asc'
    wrapper = f'<section class="pw-row-list" id="pw-row-list-{esc(rid)}">'
    if not entries:
        return f'{wrapper}<p class="sim-empty">No comparisons match the current filter.</p></section>'

    position = {id(e): i + 1 for i, e in enumerate(run.entries)}
    numbered = [(position.get(id(e), i + 1), e) for i, e in enumerate(entries)]
    if sort == 'winner':
        numbered.sort(key=lambda t: (_WINNER_ORDER.get(t[1].comparison.winner, 4), t[0]), reverse=direction == 'desc')
    else:
        numbered.sort(key=itemgetter(0), reverse=direction == 'desc')

    total = len(numbered)
    pages = max(1, -(-total // page_size))
    page = min(max(page, 1), pages)
    shown = numbered[(page - 1) * page_size : page * page_size]

    judges = judge_order(run)
    with_trace = any(single_trace_url(e.comparison.trace_id) for _, e in shown)
    head = (
        '<thead><tr>'
        f'<th>{_sort_button(rid, "#", "index", sort, direction)}</th>'
        '<th>Question</th>'
        f'<th>{_sort_button(rid, "Winner", "winner", sort, direction)}</th>'
        '<th>Judges</th>' + ('<th>Trace</th>' if with_trace else '') + '</tr></thead>'
    )
    body = ''.join(_row(rid, run, judges, n, e, with_trace=with_trace) for n, e in shown)
    return (
        f'{wrapper}<table class="pw-cmp-table">{head}{body}</table>'
        f'{_pager(rid, sort, direction, page, pages, total)}</section>'
    )


def _dot_strip(run: PairwiseRun, entries: list[PairwiseEntry]) -> str:
    judges = judge_order(run)
    position = {id(e): i + 1 for i, e in enumerate(run.entries)}
    dots = []
    for i, entry in enumerate(entries):
        n = position.get(id(entry), i + 1)
        winner = entry.comparison.winner
        split = ' is-split' if comparison_flags(entry, judges).split else ''
        colour = pw_side_colour(winner)
        dots.append(
            f'<a class="pw-dot{split}" href="#cmp-{n}" title="#{n} · {esc(pw_side_label(winner, run))}"'
            f' style="background:{colour};color:{colour}"></a>'
        )
    legend = (
        '<div class="pw-legend">'
        f'<span><span class="pw-swatch" style="background:var(--pw-a)"></span>{esc(run.label_a)} won</span>'
        f'<span><span class="pw-swatch" style="background:var(--pw-b)"></span>{esc(run.label_b)} won</span>'
        '<span><span class="pw-swatch" style="background:var(--pw-tie)"></span>tie</span>'
        '<span><span class="pw-swatch" style="background:var(--pw-none)"></span>inconclusive</span>'
        '<span>ringed = judges split · click to jump</span></div>'
    )
    return panel('Every comparison', f'<div class="pw-strip">{"".join(dots)}</div>{legend}')


def render_pairwise_comparisons(rid: str, run: PairwiseRun, entries: list[PairwiseEntry]) -> str:
    """Dot strip panel plus page 1 of the row list."""
    return _dot_strip(run, entries) + render_pairwise_row_list(rid, run, entries)


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def _vote_item(vote: PairwiseVote | None, model: str, run: PairwiseRun) -> str:
    value = vote.vote if vote else None
    head = f'{esc(_short(model))} → {esc(pw_side_label(value, run))}'
    tags = ''
    body = ''
    if vote is None:
        tags = '<span class="pw-standin">did not vote</span>'
    else:
        if vote.replacement:
            tags += '<span class="pw-standin">stand-in</span>'
        if vote.flipped:
            tags += '<span class="pw-flip">⚖ flipped when sides swapped, vote thrown out</span>'
        multi = any(o.repetition > 0 for o in vote.observations)
        lines = []
        for o in vote.observations:
            order = 'original order' if o.ordering == 'ab' else 'swapped order'
            if multi:
                order += f' · pass {o.repetition + 1}'
            lines.append(f'<div><b>{order}</b> {esc(pw_side_label(o.verdict, run))} · {esc(o.explanation or "")}</div>')
        if not lines and vote.explanation:
            lines.append(f'<div>{esc(vote.explanation)}</div>')
        body = f'<div class="pw-passes">{"".join(lines)}</div>' if lines else ''
        classify = _classify_details({'classify_observations': _classify_observations(vote, run)})
        body += classify
    return f'<li class="pw-vote"><div class="pw-vote-head" style="color:{pw_side_colour(value)}">{head}{tags}</div>{body}</li>'


def render_pairwise_comparison_detail(run: PairwiseRun, idx: int) -> str:
    """Both responses and every judge's vote for ``run.entries[idx]``."""
    entry = run.entries[idx]
    by_model = {v.model: v for v in entry.comparison.votes}
    panes = (
        '<div class="pw-panes">'
        f'<div class="pw-pane" style="--c:var(--pw-a)"><h4>{esc(run.label_a)}</h4>'
        f'<div class="pw-pane-body">{esc(entry.response_a)}</div></div>'
        f'<div class="pw-pane" style="--c:var(--pw-b)"><h4>{esc(run.label_b)}</h4>'
        f'<div class="pw-pane-body">{esc(entry.response_b)}</div></div></div>'
    )
    votes = ''.join(_vote_item(by_model.get(j), j, run) for j in judge_order(run))
    return f'{panes}<ul class="pw-votes">{votes}</ul>'


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _load_run(rid: str, roots: list[Path] | None) -> Any | None:
    from evaluatorq.dashboard import library
    from evaluatorq.dashboard.surfaces import ADAPTERS

    path = library.resolve(rid, roots)
    if path is None:
        return None
    surface, _raw = library.load_surface(path)
    if surface != 'pairwise':
        return None
    return ADAPTERS['pairwise'].load(path)


def _404() -> Response:
    return Response('404 Not Found', 404, media_type='text/plain')


def register_pairwise_view_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register the pairwise comparisons fragment routes on *app*."""

    @app.get('/r/{rid}/pairwise/row-list')
    def pairwise_row_list(rid: str, req: Request) -> Response:
        run = _load_run(rid, roots)
        if run is None:
            return _404()
        entries = apply_or_all(run, 'pairwise', parse_selections(req, 'pairwise'))
        try:
            page = int(req.query_params.get('page', '1'))
        except (ValueError, TypeError):
            page = 1
        html = render_pairwise_row_list(
            rid,
            run,
            entries,
            sort=req.query_params.get('sort', 'index'),
            direction=req.query_params.get('dir', 'asc'),
            page=page,
        )
        return Response(html, media_type='text/html')

    @app.get('/r/{rid}/pairwise/comparison')
    def pairwise_comparison(rid: str, req: Request) -> Response:
        run = _load_run(rid, roots)
        if run is None:
            return _404()
        try:
            idx = int(req.query_params.get('idx', ''))
        except (ValueError, TypeError):
            idx = -1
        if not 0 <= idx < len(run.entries):
            return Response('<p class="sim-empty">No comparison at that index.</p>', media_type='text/html')
        return Response(render_pairwise_comparison_detail(run, idx), media_type='text/html')
