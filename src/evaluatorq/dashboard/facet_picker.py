"""The filter picker shared by Traces, Trace search and Insights: a two-level facet menu and its chips."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc
from evaluatorq.trace_finder.models import FACET_NAMES, NUMERIC_FACET_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection, NumericFilters

FACET_LABELS = (
    ('project', 'project'),
    ('agent_name', 'agent'),
    ('model', 'model'),
    ('provider', 'provider'),
    ('status', 'status'),
    ('product', 'product'),
    ('trace_type', 'trace type'),
    ('tool_name', 'tool'),
    ('tokens', 'total tokens'),
    ('duration_ms', 'duration'),
)
if {name for name, _ in FACET_LABELS} != {*FACET_NAMES, *NUMERIC_FACET_NAMES}:
    raise RuntimeError('FACET_LABELS must mirror FACET_NAMES and NUMERIC_FACET_NAMES')


def window_count_note(window_days: int | None) -> str:
    """Say what an Orq per-value count covers: the traces in the catalogue's time window."""
    if window_days is None:
        return 'Traces in the selected time window'
    return f'Traces in the last {window_days:,} day{"" if window_days == 1 else "s"}'


def _facet_values(catalogue: FacetCatalogue | None, name: str) -> tuple[str, ...]:
    if catalogue is None or name in NUMERIC_FACET_NAMES:
        return ()
    return tuple(getattr(catalogue, name, ()))


def _counted_values(values: tuple[str, ...], counts: Mapping[str, int]) -> tuple[tuple[str, ...], dict[str, int]]:
    """Order values by count, most first, and resolve each one's count.

    A value matches its exact key first, then any key equal ignoring case. Count keys no value matches
    are appended, so values only the counts carry still show. ``sorted`` is stable, so ties keep the
    catalogue order.
    """
    folded: dict[str, int] = {}
    for key, count in counts.items():
        folded[key.casefold()] = folded.get(key.casefold(), 0) + count
    known = {value.casefold() for value in values}
    values = (*values, *(key for key in counts if key.casefold() not in known))
    resolved = {value: counts[value] if value in counts else folded.get(value.casefold(), 0) for value in values}
    return tuple(sorted(values, key=lambda value: -resolved[value])), resolved


def render_facet_menu(
    catalogue: FacetCatalogue | None,
    *,
    form_id: str,
    selection: FacetSelection | None = None,
    numeric: NumericFilters | None = None,
    include_numeric: bool = True,
    counts: Mapping[str, Mapping[str, int]] | None = None,
    count_note: str = '',
    count_title: str = '',
    list_note_html: str = '',
    container_attrs: str = '',
    open_: bool = False,
) -> str:
    """Two-level filter menu: a category list, and a value pop-out the client opens per category.

    ``counts`` maps a facet to a count per value: each value shows its count, values sort by it, and a
    value without an entry shows 0. A facet missing from ``counts`` shows no counts. ``count_note`` is
    plain text saying what the number counts; ``count_title`` is the badge tooltip and defaults to the
    note. ``list_note_html`` and ``container_attrs`` are trusted, caller-built HTML.
    """
    title = count_title or count_note
    title_attr = f' title="{esc(title)}"' if title else ''
    items: list[str] = []
    subs: list[str] = []
    for name, label in FACET_LABELS:
        numeric_facet = name in NUMERIC_FACET_NAMES
        if numeric_facet and not include_numeric:
            continue
        selected_values = getattr(selection, name, frozenset()) if selection is not None else frozenset()
        values = tuple(dict.fromkeys((*_facet_values(catalogue, name), *sorted(selected_values))))
        if numeric_facet:
            minimum = getattr(numeric, f'{name}_min', None) if numeric is not None else None
            maximum = getattr(numeric, f'{name}_max', None) if numeric is not None else None
            count = sum(bound is not None for bound in (minimum, maximum))
            body = (
                f'<label><span>≥</span><input form="{form_id}" name="{esc(name)}_min" type="number" min="0" '
                f'placeholder="min" value="{esc(str(minimum)) if minimum is not None else ""}"></label>'
                f'<label><span>≤</span><input form="{form_id}" name="{esc(name)}_max" type="number" min="0" '
                f'placeholder="max" value="{esc(str(maximum)) if maximum is not None else ""}"></label>'
            )
        else:
            count = len(selected_values)
            facet_counts = counts.get(name) if counts is not None else None
            resolved: dict[str, int] | None = None
            if facet_counts is not None:
                values, resolved = _counted_values(values, facet_counts)
            options = ''.join(
                f'<label><input form="{form_id}" type="checkbox" name="facet_{esc(name)}" value="{esc(value)}"{(" checked" if value in selected_values else "")}><span>{esc(value)}</span>'
                + (f'<span class="facet-n"{title_attr}>{resolved[value]:,}</span>' if resolved is not None else '')
                + '</label>'
                for value in values
            )
            if options:
                overflow = (
                    '<p class="facet-note">More values exist in Orq; showing the returned values ranked by frequency.</p>'
                    if catalogue is not None and name in catalogue.truncated_facets
                    else ''
                )
                scope = f'<p class="facet-scope">{esc(count_note)}</p>' if resolved is not None and count_note else ''
                body = (
                    f'{scope}<input class="facet-search" type="search" placeholder="Search values" aria-label="Search {esc(label)} values" autocomplete="off">'
                    f'<div class="facet-values">{options}</div>'
                    '<p class="facet-no-results" hidden>No matching values in this list.</p>'
                    f'{overflow}'
                )
            else:
                body = '<p class="finder-empty">No values in this window.</p>'
        count_html = f'<span class="count">{count}</span>' if count else ''
        items.append(
            f'<button type="button" class="facet-item" data-facet="{esc(name)}" aria-haspopup="true" '
            f'aria-expanded="false"><span>{esc(label)}</span>{count_html}<span class="chev" aria-hidden="true">&rsaquo;</span></button>'
        )
        subs.append(
            f'<div class="facet-sub" data-facet-sub="{esc(name)}" hidden><div class="hd">{esc(label)}</div>{body}</div>'
        )
    return (
        f'<div class="finder-facets{" open" if open_ else ""}"{container_attrs}><div class="facet-list">{"".join(items)}{list_note_html}</div>'
        f'{"".join(subs)}</div>'
    )


def render_facet_chips(
    selection: FacetSelection,
    numeric: NumericFilters | None = None,
    *,
    removable: bool = False,
    generated: FacetSelection | None = None,
) -> str:
    """Render one chip per active filter value. Editable chips open the already-rendered menu at their category."""

    def chip(
        facet: str,
        label: str,
        value_html: str,
        remove_name: str,
        remove_value: str | None,
        aria: str,
        *,
        ai: bool = False,
    ) -> str:
        ai_class = ' ai' if ai else ''
        badge = '<b class="ai-badge">AI</b>' if ai else ''
        if not removable:
            return f'<span class="chip{ai_class}"><b>{esc(label)}</b>{badge}<span class="v">{value_html}</span></span>'
        value_attr = f' data-finder-value="{esc(remove_value)}"' if remove_value is not None else ''
        return (
            f'<span class="chip is-editable{ai_class}" data-chip-name="{esc(remove_name)}"{value_attr}>'
            f'<button type="button" class="chip-open" data-chip-open="{esc(facet)}" aria-label="Edit {aria}">'
            f'<b>{esc(label)}</b>{badge}<span class="v">{value_html}</span></button>'
            f'<button type="button" class="finder-chip-remove" data-finder-remove="{esc(remove_name)}"{value_attr} '
            f'aria-label="Remove {aria}">✕</button></span>'
        )

    chips: list[str] = []
    for name, label in FACET_LABELS:
        if name in NUMERIC_FACET_NAMES:
            continue
        chips.extend(
            chip(
                name,
                label,
                esc(value),
                f'facet_{name}',
                value,
                f'{esc(label)} {esc(value)}',
                ai=bool(generated and value in getattr(generated, name)),
            )
            for value in sorted(getattr(selection, name, frozenset()))
        )
    for name, label, operator in (
        ('tokens_min', 'total tokens', '≥'),
        ('tokens_max', 'total tokens', '≤'),
        ('duration_ms_min', 'duration', '≥'),
        ('duration_ms_max', 'duration', '≤'),
    ):
        value = getattr(numeric, name, None) if numeric is not None else None
        if value is None:
            continue
        facet = name.rsplit('_', 1)[0]
        formatted = f'{operator} {value:,}'
        chips.append(chip(facet, label, formatted, name, None, f'{label} {formatted}'))
    return ''.join(chips)
