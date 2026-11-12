"""Table state, view/sort constants and compare URL handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.filters import FILTERS

if TYPE_CHECKING:
    from evaluatorq.dashboard.sim_compare.model import Comparison


def _qs(cmp: Comparison, **params: Any) -> str:
    return urlencode({'a': cmp.a.rid, 'b': cmp.b.rid, **params})


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
    ('score', 'Goal score A → B', lambda pc: pc[0].b.goal_completion_score - pc[0].a.goal_completion_score),
    ('turns', 'Turns A → B', lambda pc: pc[0].b.turn_count - pc[0].a.turn_count),
    (
        'change',
        'Change',
        lambda pc: (_SEVERITY[pc[1]], pc[0].b.goal_completion_score - pc[0].a.goal_completion_score),
    ),
]
_SORT_KEYS = {param: key for param, _, key in _COLUMNS}
_DEFAULT_SORT = 'change'  # regressions first, then by the biggest score drop
_DIRECTION_LABELS = {'change': ('worst first', 'best first')}
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


def canonical_url(cmp: Comparison, state: TableState) -> str:
    """The GET ``/compare/sim`` URL that reproduces this slice and table state.

    Selections use the repeated ``dimension=value`` form ``parse_selections``
    reads, limited to filters that drop a conversation; table params equal to
    their defaults are left out.
    """
    return _compare_url(cmp, cmp.narrowing, state)


def _compare_url(cmp: Comparison, narrowing: dict[str, list[str]], state: TableState) -> str:
    params: list[tuple[str, str]] = [('a', cmp.a.rid), ('b', cmp.b.rid)]
    params.extend((dim, value) for dim, values in narrowing.items() for value in values)
    table = (
        ('cmp_view', state.view, ''),
        ('cmp_sort', state.sort, _DEFAULT_SORT),
        ('cmp_dir', state.direction, 'asc'),
        ('cmp_page', state.page, 1),
        ('cmp_size', state.size, _DEFAULT_SIZE),
    )
    params.extend((name, str(value)) for name, value, default in table if value != default)
    return f'/compare/sim?{urlencode(params)}'


def _swap_link(cmp: Comparison, narrowing: dict[str, list[str]], state: TableState) -> str:
    """``href`` + HTMX attributes that load the compare view for *narrowing*
    through the filter route, swapping the page and rail like a rail change."""
    url = _compare_url(cmp, narrowing, state)
    query = url.partition('?')[2]
    return f'href="{esc(url)}" hx-get="/compare/sim/filter?{esc(query)}" hx-target="#filter-swap" hx-swap="outerHTML"'


def _base_state(state: TableState | None) -> TableState:
    """*state* back on page 1, which is where any filter change lands."""
    state = state or TableState()
    return TableState(state.view, state.sort, state.direction, 1, state.size)


def _toggled(narrowing: dict[str, list[str]], dim: str, value: str) -> dict[str, list[str]]:
    """*narrowing* with *value* added to *dim*, or removed when already there."""
    values = narrowing.get(dim, [])
    rest = [v for v in values if v != value]
    out = {k: v for k, v in narrowing.items() if k != dim}
    if value in values:
        return {**out, **({dim: rest} if rest else {})}
    return {**out, dim: [*values, value]}


def parse_compare_params(items: list[tuple[str, str]]) -> tuple[dict[str, list[str]], TableState]:
    """Split request parameters into the sim filter selections and the table state.

    Every compare route reads its parameters here, from the query string or the
    rail form alike, so a parameter means the same thing on each.
    """
    dims = set(FILTERS['sim'].dimensions)
    selections: dict[str, list[str]] = {}
    table: dict[str, str] = {}
    for key, value in items:
        if key.startswith('cmp_'):
            table[key] = value
        elif key in dims:
            selections.setdefault(key, []).append(value)
    return selections, TableState.parse(table)
