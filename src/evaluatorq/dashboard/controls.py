"""Shared filter controls for the dashboard: rich picker, dropdown, chip, range slider and stat tile.

The filter rails (sim, red team, pairwise), Settings and the Pareto Frontier
toolbar build their inputs from these, so a control looks and behaves the same
wherever it appears.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc

if TYPE_CHECKING:
    from collections.abc import Sequence


def stat_tile(label: str, value: str, unit: str = '', sub: str = '') -> str:
    unit_html = f'<span class="stat-unit">{esc(unit)}</span>' if unit else ''
    sub_html = f'<div class="stat-sub">{esc(sub)}</div>' if sub else ''
    return (
        f'<div class="stat-tile"><div class="stat-label">{esc(label)}</div>'
        f'<div class="stat-value">{esc(value)}{unit_html}</div>{sub_html}</div>'
    )


def rich_pick(
    name: str,
    value: str,
    options: Sequence[tuple[str, str]],
    *,
    disabled: Sequence[str] = (),
    current: str = '',
    describedby: str = '',
    compact: bool = False,
    filterable: bool = False,
) -> str:
    """A menu of rich rows that submits *value* through a hidden input.

    Reuses the model picker's markup and script; ``data-rich`` makes the trigger
    copy the chosen row, so the closed control keeps its note and badge.
    *disabled* rows are listed but cannot be chosen. The trigger shows the row
    matching *value*, else *current*. *compact* puts each row's note on the
    title's line, for long lists.
    """
    selected = next((row for option, row in options if value and option.rstrip('/') == value.rstrip('/')), current)
    buttons = ''.join(
        f'<button type="button" class="model-option{" is-selected" if row == selected else ""}" data-model="{esc(option)}"'
        f' data-rich aria-pressed="{"true" if row == selected else "false"}">{row}</button>'
        for option, row in options
    )
    rows_off = ''.join(f'<div class="rich-pick-disabled" aria-disabled="true">{row}</div>' for row in disabled)
    described = f' aria-describedby="{esc(describedby)}"' if describedby else ''
    search_html = (
        f'<input class="facet-search" type="search" placeholder="Search" aria-label="Search {esc(name.replace("_", " "))} options" autocomplete="off">'
        '<div class="facet-values">'
        if filterable
        else ''
    )
    empty_html = '</div><p class="facet-no-results" hidden>No matches.</p>' if filterable else ''
    return (
        f'<span class="model-pick rich-pick{" rich-pick--compact" if compact else ""}{" rich-pick--filter" if filterable else ""}">'
        f'<input type="hidden" name="{esc(name)}" value="{esc(value)}">'
        f'<button type="button" id="{esc(name)}" class="model-pick-btn" aria-haspopup="true" aria-expanded="false"{described}>'
        f'{selected}</button>'
        f'<div class="finder-facets"><div class="facet-list">'
        f'{search_html}{buttons}{rows_off}{empty_html}'
        f'</div></div></span>'
    )


FILTER_CHEVRON = (
    '<svg class="filter-dd-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="m6 9 6 6 6-6"/></svg>'
)


def _filter_input_id(name: str, value: str) -> str:
    """Stable per (name, value) input id so htmx restores focus to the same control after a swap."""
    return f'fc-{name}-{hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest()[:8]}'


def chip(name: str, value: str, *, checked: bool, dot_cls: str, label: str | None = None) -> str:
    """One chip toggle: a <label>-wrapped visually-hidden checkbox.

    *label* is the visible text; it defaults to *value* (the submitted value).
    """
    checked_attr = ' checked' if checked else ''
    active_cls = ' is-active' if checked else ''
    chip_id = _filter_input_id(name, value)
    return (
        f'<label class="filter-chip{active_cls}">'
        f'<input type="checkbox" id="{esc(chip_id)}" class="filter-chip-input" name="{esc(name)}"'
        f' value="{esc(value)}"{checked_attr}>'
        f'<span class="filter-chip-dot {dot_cls}" aria-hidden="true"></span>'
        f'<span class="filter-chip-check {dot_cls}" aria-hidden="true">✓</span>'
        f'<span class="filter-chip-label">{esc(value if label is None else label)}</span>'
        f'</label>'
    )


def dropdown(dim: str, label: str, dim_opts: list[str], sel: list[str], *, searchable: bool = False) -> str:
    """One Persona/Scenario <details> dropdown with a stable id.

    ``sel`` empty means "all selected" (the default filter-route behaviour);
    the trigger status/label reflect that. *searchable* adds a search box above
    the rows that ``dashboard.js`` uses to hide the rows that do not match.
    """
    selected_set = set(sel) if sel else set(dim_opts)
    n_selected = len(selected_set)
    n_total = len(dim_opts)
    if n_selected == n_total:
        status_cls, status_label, value_cls = 'is-all', 'All', 'filter-dd-value'
    elif n_selected == 0:
        status_cls, status_label, value_cls = 'is-none', 'None', 'filter-dd-value is-none'
    else:
        status_cls, status_label, value_cls = 'is-partial', f'{n_selected} selected', 'filter-dd-value is-engaged'

    rows = ''.join(
        f'<label class="filter-dd-row">'
        f'<input type="checkbox" id="{esc(_filter_input_id(dim, opt))}" name="{esc(dim)}" value="{esc(opt)}"'
        f'{" checked" if opt in selected_set else ""}>'
        f'<span>{esc(opt)}</span>'
        f'</label>'
        for opt in dim_opts
    )
    if searchable:
        rows = (
            f'<input class="facet-search" type="search" placeholder="Search"'
            f' aria-label="Search {esc(label.lower())} options" autocomplete="off">'
            f'<div class="facet-values">{rows}</div>'
            f'<p class="facet-no-results" hidden>No matches.</p>'
        )
    return (
        f'<details id="filter-dd-{esc(dim)}" class="filter-dd">'
        f'<summary class="filter-dd-trigger">'
        f'<span class="filter-dd-status {status_cls}"></span>'
        f'<span class="filter-dd-name">{esc(label)}</span>'
        f'<span class="{value_cls}">{esc(status_label)}</span>'
        f'{FILTER_CHEVRON}'
        f'</summary>'
        f'<div class="filter-dd-menu">{rows}</div>'
        f'</details>'
    )


def _fmt_range_value(value: float, step: str) -> str:
    """Format a range input's numeric attribute: integers for raw counts
    (``step="1"``), compact decimals for score thresholds (``step="0.05"``)."""
    if step == '1':
        return str(int(value))
    return f'{value:g}'


def range_engaged(sel: list[str], *, min_val: float, max_val: float, floor: bool) -> bool:
    """True when a range control is actively constraining results, i.e. its
    value differs from the no-op default bound (``min_val`` for floors,
    ``max_val`` for ceilings). Absent/unparseable selections are not engaged."""
    if not sel:
        return False
    try:
        value = float(sel[0])
    except (ValueError, TypeError):
        return False
    return value != (min_val if floor else max_val)


def range_control(
    dim: str,
    label: str,
    *,
    min_val: float,
    max_val: float,
    step: str,
    sel: list[str],
    floor: bool,
    show_max: bool = False,
) -> str:
    """One min/max range control shared by raw-count and score thresholds.

    ``floor`` selects both the unset ("no-op") end of the range — ``min_val``
    for a floor (e.g. min turns), ``max_val`` for a ceiling (e.g. max goal
    score) — and the readout glyph (``≥`` for floors, ``≤`` for ceilings).
    An absent selection renders the default bound numerically (never "all").
    ``show_max`` appends the slider's upper bound beside the readout (e.g.
    min-turns shows the run's max turns). The readout gains ``is-engaged`` when
    the slider is moved off its no-op bound, so an active filter reads at a
    glance without comparing numbers.
    """
    raw = sel[0] if sel else None
    default = min_val if floor else max_val
    try:
        value = float(raw) if raw is not None else default
    except (ValueError, TypeError):
        value = default
    engaged = range_engaged(sel, min_val=min_val, max_val=max_val, floor=floor)
    readout_cls = 'filter-slider-readout is-engaged' if engaged else 'filter-slider-readout'
    glyph = '≥' if floor else '≤'
    readout = f'{glyph} {_fmt_range_value(value, step)}'
    max_span = f'<span class="filter-slider-max">/ {_fmt_range_value(max_val, step)}</span>' if show_max else ''
    # data-glyph / data-default let dashboard.js update the readout live on
    # `input` (while dragging), before the HTMX `change` round-trip.
    return (
        f'<div class="filter-group" data-dim="{esc(dim)}">'
        f'<label class="filter-label">{esc(label)}</label>'
        f'<div class="filter-slider-row">'
        f'<input type="range" class="filter-slider" name="{esc(dim)}" min="{_fmt_range_value(min_val, step)}"'
        f' max="{_fmt_range_value(max_val, step)}" step="{esc(step)}" value="{_fmt_range_value(value, step)}"'
        f' data-glyph="{glyph}" data-default="{_fmt_range_value(default, step)}">'
        f'<span class="{readout_cls}">{esc(readout)}</span>'
        f'{max_span}'
        f'</div>'
        f'</div>'
    )
