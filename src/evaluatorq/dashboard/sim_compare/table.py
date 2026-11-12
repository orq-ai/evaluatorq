"""The compare conversation table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.sim_compare.fmt import _badge, _change_pill, _run_key, _tone
from evaluatorq.dashboard.sim_compare.state import (
    _COLUMNS,
    _DIRECTION_LABELS,
    _PAGE_SIZES,
    _SORT_KEYS,
    _VIEWS,
    TableState,
    _base_state,
    _qs,
    _swap_link,
)
from evaluatorq.dashboard.sim_views import (
    _sim_header_row,
    _sim_pager,
)

if TYPE_CHECKING:
    from evaluatorq.dashboard.sim_compare.model import Comparison
    from evaluatorq.simulation.contracts import SimulationEntry


def _in_view(change: str, view: str) -> bool:
    if view == 'all':
        return True
    if view == 'changed':
        return change != 'same'
    return change == view


def resolve_view(cmp: Comparison, state: TableState) -> str:
    """The segment the table shows. Defaults to the changed conversations — the
    reason to open a comparison — unless nothing changed, then everything."""
    if state.view:
        return state.view
    changed = any(c != 'same' for _, c in cmp.changes)
    return 'changed' if changed else 'all'


def _empty(message: str, cmp: Comparison, state: TableState, *, show_all: Any = None) -> str:
    """The empty table, with a way out: clear the filters when any are active,
    and (given *show_all*, the segment control builder) widen the segment to all."""
    actions = []
    if cmp.narrowing:
        actions.append(f'<a class="cmp-empty-btn" {_swap_link(cmp, {}, _base_state(state))}>Clear all filters</a>')
    if show_all and resolve_view(cmp, state) != 'all':
        attrs = show_all(state.sort, state.direction, 1, state.size, v='all')
        actions.append(f'<button type="button" class="cmp-empty-btn" {attrs}>Show all conversations</button>')
    buttons = f'<div class="cmp-empty-actions">{"".join(actions)}</div>' if actions else ''
    return f'<div class="cmp-empty"><p class="sim-empty">{message}</p>{buttons}</div>'


def render_conversations(cmp: Comparison, state: TableState) -> str:
    """The conversation section: segment control, sortable/paged table whose
    rows open the paired transcript in the drawer, and the unmatched lists."""
    changes = cmp.changes
    counts = {key: sum(1 for _, c in changes if _in_view(c, key)) for key, _ in _VIEWS}
    view = resolve_view(cmp, state)

    def control(sort: str, direction: str, page: int, size: int, *, v: str = state.view) -> str:
        qs = _qs(cmp, cmp_view=v, cmp_sort=sort, cmp_dir=direction, cmp_page=page, cmp_size=size)
        return (
            f'hx-get="/compare/sim/conversations?{esc(qs)}" hx-include="#filter-form" '
            'hx-target="#cmp-conversations" hx-swap="outerHTML"'
        )

    seg = ''.join(
        (
            f'<button type="button" class="cmp-seg-btn" aria-pressed="true">{label}'
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
            f'<tr class="sim-conv-row cmp-conv-row" role="button" tabindex="0" '
            f'data-sim-entity-trigger data-entity-kind="conversation" data-drawer-url="{esc(url)}">'
            f'<td class="sim-conv-persona">{esc(a.persona)}</td>'
            f'<td class="sim-conv-scenario">{esc(a.scenario)}</td>'
            f'<td class="cmp-num">{a.goal_completion_score:.2f} &rarr; {b.goal_completion_score:.2f}'
            f' <span class="cmp-num-d cmp-num-d--{_tone(d_score)}">{d_score:+.2f}</span></td>'
            f'<td class="cmp-num">{a.turn_count} &rarr; {b.turn_count} <span class="cmp-num-d">{d_turns:+d}</span></td>'
            f'<td>{_change_pill(change)}</td>'
            f'<td class="cmp-outcome">{_badge(cmp.a.status(a))}<span class="cmp-arrow">&rarr;</span>{_badge(cmp.b.status(b))}</td>'
            '</tr>'
        )

    if rows:
        header = _sim_header_row(
            control,
            state.sort,
            state.direction,
            state.size,
            columns=_COLUMNS,
            trailing=('Outcome A → B',),
            direction_labels=_DIRECTION_LABELS,
        )
        table = (
            '<div class="sim-conv-table-shell">'
            f'<table class="sim-conv-table cmp-conv-table">{header}<tbody>{"".join(rows)}</tbody></table>'
            '</div>'
            f'{_sim_pager(control, state.sort, state.direction, page_no, pages, total, state.size, noun="pairs", sizes=_PAGE_SIZES)}'
        )
    elif not cmp.matching.matched:
        table = _empty('No conversations matched on (persona, scenario).', cmp, state)
    else:
        table = _empty(f'No {view} conversations under the current filters.', cmp, state, show_all=control)

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
            ('cmp_view', state.view),
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
