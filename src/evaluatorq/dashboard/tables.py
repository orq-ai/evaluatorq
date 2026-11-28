"""Shared pieces of the dashboard's sortable tables."""

from __future__ import annotations

from evaluatorq.common.reports import esc


def flipped(direction: str) -> str:
    """The other sort direction: ``'asc'`` for ``'desc'`` and the reverse."""
    return 'desc' if direction == 'asc' else 'asc'


def sortable_th(
    label: str,
    *,
    control: str,
    active: bool,
    direction: str,
    numeric: bool = False,
    word: tuple[str, str] | None = None,
    tag: str = 'button',
) -> str:
    """A sortable ``<th>``: `label` in a clickable `tag` whose opening attributes are `control`.

    `direction` is the order the column is sorted in now, ``'asc'`` or ``'desc'``; it only shows, and sets
    ``aria-sort``, when `active`. Inactive columns show a faint up-down glyph so the header reads as
    clickable. `word` is an (asc, desc) pair shown instead of the arrow for an order that is not low-to-high
    (a severity sort reads "worst first", not "ascending"). `numeric` right-aligns the header over its cells.
    """
    aria = {'asc': 'ascending', 'desc': 'descending'}[direction] if active else 'none'
    if active and word:
        caret, caret_cls = esc(word[0 if direction == 'asc' else 1]), 'th-caret th-caret-active th-caret-word'
    elif active:
        caret, caret_cls = ('&#x25B2;' if direction == 'asc' else '&#x25BC;'), 'th-caret th-caret-active'
    else:
        caret, caret_cls = '&#x21C5;', 'th-caret'
    num = ' class="num"' if numeric else ''
    return (
        f'<th scope="col" aria-sort="{aria}"{num}><{tag} class="th-sort" {control}>'
        f'{esc(label)}<span class="{caret_cls}">{caret}</span></{tag}></th>'
    )
