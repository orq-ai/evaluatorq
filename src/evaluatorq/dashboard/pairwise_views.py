"""Pairwise comparisons list: dot strip, paged row list and lazy detail rows.

Routes (HTML fragments, no page shell):

    GET /r/{rid}/pairwise/row-list   → filtered, sorted, paged row list; ``focus=N``
                                       (1-based full-run number) shows the page holding
                                       comparison N with that row open
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
from evaluatorq.dashboard.orq_workspace import cli_slug_render_scope
from evaluatorq.dashboard.pairwise_palette import pw_side_colour, pw_side_label
from evaluatorq.dashboard.report_kit import panel
from evaluatorq.dashboard.sim_compare import resolve_run
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


def _row(
    rid: str,
    run: PairwiseRun,
    judges: list[str],
    n: int,
    entry: PairwiseEntry,
    *,
    with_trace: bool,
    is_open: bool,
) -> str:
    winner = entry.comparison.winner
    trace = ''
    if with_trace:
        trace = trace_link_button(
            single_trace_url(entry.comparison.trace_id),
            'Trace ↗',
            onclick='event.stopPropagation()',
        )
    trace_cell = f'<span class="pw-cmp-trace">{trace}</span>' if with_trace else ''
    summary = (
        '<summary class="pw-cmp-summary">'
        f'<span class="pw-cmp-n">{n}</span>'
        f'<span class="pw-cmp-q">{esc(entry.question)}</span>'
        f'<span class="pw-cmp-win"><span class="pw-vchip" style="--c:{pw_side_colour(winner)}">'
        f'{esc(pw_side_label(winner, run))}</span></span>'
        f'<span class="pw-chiprow">{_chips(entry, run, judges)}</span>'
        f'{trace_cell}'
        '</summary>'
    )
    # A focused row arrives open with its detail already rendered: the ``toggle``
    # a parsed ``open`` fires lands before htmx has bound the lazy trigger.
    if is_open:
        return (
            f'<tbody class="pw-cmp-group" id="cmp-{n}"><tr><td><details class="pw-cmp-row" open>'
            f'{summary}<div class="pw-cmp-detail">{render_pairwise_comparison_detail(run, n - 1)}</div>'
            '</details></td></tr></tbody>'
        )
    return (
        f'<tbody class="pw-cmp-group" id="cmp-{n}"><tr><td>'
        '<details class="pw-cmp-row"'
        f' hx-get="/r/{esc(rid)}/pairwise/comparison?idx={n - 1}"'
        ' hx-trigger="toggle once" hx-target="find .pw-cmp-detail" hx-swap="innerHTML">'
        f'{summary}<div class="pw-cmp-detail"></div></details></td></tr></tbody>'
    )


def _numbers(run: PairwiseRun, entries: list[PairwiseEntry]) -> list[tuple[int, PairwiseEntry]]:
    """``(1-based full-run number, entry)`` pairs; numbering survives filtering."""
    position = {id(e): i + 1 for i, e in enumerate(run.entries)}
    return [(position.get(id(e), i + 1), e) for i, e in enumerate(entries)]


def valid_focus(run: PairwiseRun, entries: list[PairwiseEntry], focus: int | None) -> int | None:
    """``focus`` when it numbers one of ``entries``, else ``None``."""
    if focus is None:
        return None
    return focus if any(n == focus for n, _ in _numbers(run, entries)) else None


def render_pairwise_row_list(
    rid: str,
    run: PairwiseRun,
    entries: list[PairwiseEntry],
    *,
    sort: str = 'index',
    direction: str = 'asc',
    page: int = 1,
    page_size: int = _PAGE_SIZE,
    focus: int | None = None,
) -> str:
    """Render the paged row list for ``entries`` (a subset of ``run.entries``).

    A ``focus`` that numbers one of ``entries`` selects the page holding it (``page``
    is then ignored) and renders that row open; any other ``focus`` is ignored.
    """
    sort = sort if sort in _SORTS else 'index'
    direction = direction if direction in ('asc', 'desc') else 'asc'
    wrapper = f'<section class="pw-row-list" id="pw-row-list-{esc(rid)}">'
    if not entries:
        return f'{wrapper}<p class="sim-empty">No comparisons match the current filter.</p></section>'

    numbered = _numbers(run, entries)
    if sort == 'winner':
        numbered.sort(key=lambda t: (_WINNER_ORDER.get(t[1].comparison.winner, 4), t[0]), reverse=direction == 'desc')
    else:
        numbered.sort(key=itemgetter(0), reverse=direction == 'desc')

    total = len(numbered)
    pages = max(1, -(-total // page_size))
    focused = next((i for i, (n, _) in enumerate(numbered) if n == focus), None) if focus is not None else None
    if focused is not None:
        page = focused // page_size + 1
    page = min(max(page, 1), pages)
    shown = numbered[(page - 1) * page_size : page * page_size]

    judges = judge_order(run)
    with_trace = any(single_trace_url(e.comparison.trace_id) for _, e in shown)
    trace_head = '<span>Trace</span>' if with_trace else ''
    head = (
        '<thead><tr><th><div class="pw-cmp-head">'
        f'{_sort_button(rid, "#", "index", sort, direction)}'
        '<span>Question</span>'
        f'{_sort_button(rid, "Winner", "winner", sort, direction)}'
        f'<span>Judges</span>{trace_head}'
        '</div></th></tr></thead>'
    )
    body = ''.join(
        _row(rid, run, judges, n, e, with_trace=with_trace, is_open=focused is not None and n == focus)
        for n, e in shown
    )
    table_class = 'pw-cmp-list has-trace' if with_trace else 'pw-cmp-list'
    return (
        f'{wrapper}<table class="{table_class}">{head}{body}</table>'
        f'{_pager(rid, sort, direction, page, pages, total)}</section>'
    )


def _dot_strip(rid: str, run: PairwiseRun, entries: list[PairwiseEntry]) -> str:
    judges = judge_order(run)
    dots = []
    for n, entry in _numbers(run, entries):
        winner = entry.comparison.winner
        split = ' is-split' if comparison_flags(entry, judges).split else ''
        colour = pw_side_colour(winner)
        # The row may be on another page, sorted away or not rendered yet: fetch the
        # list focused on it, then scroll to it. The href is the no-JS fallback.
        dots.append(
            f'<a class="pw-dot{split}" href="#cmp-{n}" title="#{n} · {esc(pw_side_label(winner, run))}"'
            f' hx-get="/r/{esc(rid)}/pairwise/row-list?focus={n}" hx-include="#filter-form"'
            f' hx-target="#pw-row-list-{esc(rid)}" hx-swap="outerHTML show:#cmp-{n}:top"'
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


def render_pairwise_comparisons(
    rid: str, run: PairwiseRun, entries: list[PairwiseEntry], *, focus: int | None = None
) -> str:
    """Dot strip panel plus the row list (page 1, or the page holding ``focus``)."""
    return _dot_strip(rid, run, entries) + render_pairwise_row_list(rid, run, entries, focus=focus)


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


def _404() -> Response:
    return Response('404 Not Found', 404, media_type='text/plain')


def register_pairwise_view_routes(app: Any, roots: list[Path] | None = None) -> None:
    """Register the pairwise comparisons fragment routes on *app*."""

    @app.get('/r/{rid}/pairwise/row-list')
    def pairwise_row_list(rid: str, req: Request) -> Response:
        _, run = resolve_run(rid, roots, 'pairwise')
        if run is None:
            return _404()
        entries = apply_or_all(run, 'pairwise', parse_selections(req, 'pairwise'))
        try:
            page = int(req.query_params.get('page', '1'))
        except (ValueError, TypeError):
            page = 1
        try:
            focus: int | None = int(req.query_params.get('focus', ''))
        except (ValueError, TypeError):
            focus = None
        with cli_slug_render_scope():
            html = render_pairwise_row_list(
                rid,
                run,
                entries,
                sort=req.query_params.get('sort', 'index'),
                direction=req.query_params.get('dir', 'asc'),
                page=page,
                focus=focus,
            )
        return Response(html, media_type='text/html')

    @app.get('/r/{rid}/pairwise/comparison')
    def pairwise_comparison(rid: str, req: Request) -> Response:
        _, run = resolve_run(rid, roots, 'pairwise')
        if run is None:
            return _404()
        try:
            idx = int(req.query_params.get('idx', ''))
        except (ValueError, TypeError):
            idx = -1
        if not 0 <= idx < len(run.entries):
            return Response('<p class="sim-empty">No comparison at that index.</p>', media_type='text/html')
        return Response(render_pairwise_comparison_detail(run, idx), media_type='text/html')
