"""The "Where it changed" persona x scenario heatmap of goal-rate change from A to B."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.report_kit import _interp_color, heatmap, panel
from evaluatorq.dashboard.sim_compare.state import TableState, first_page, swap_link

if TYPE_CHECKING:
    from evaluatorq.dashboard.sim_compare.model import Comparison, MatchedPair

# Most rows and columns drawn; the rest collapse into a "+N more" note.
_MAX_AXIS = 12

# Diverging fill over a goal-rate change of -100 to +100 pts: red worse, neutral flat, green better.
# Both ends stay light enough for dark ink. The legend is drawn from the same stops.
_WORSE, _FLAT_FILL, _BETTER = '#e58f89', '#eceff1', '#7fc8a5'
_DIVERGING = ((0.0, _WORSE), (0.5, _FLAT_FILL), (1.0, _BETTER))

_MINUS = '\N{MINUS SIGN}'


def _cell_value(cmp: Comparison, pairs: list[tuple[MatchedPair, str]]) -> tuple[str, str, str]:
    """``(text, fill, title)`` for the pairs of one cell.

    The value is B's goal rate minus A's over the pairs where neither side
    errored, the same ``Side.status`` that ``pair_change`` reads; error pairs
    are counted in the title and never as misses. A cell with no such pair
    reads "err".
    """
    kinds = Counter(change for _, change in pairs)
    scored = [
        (cmp.a.status(p.a) == 'met', cmp.b.status(p.b) == 'met')
        for p, _ in pairs
        if 'error' not in (cmp.a.status(p.a), cmp.b.status(p.b))
    ]
    n = len(pairs)
    title = f'{n} pair{"" if n == 1 else "s"}: {kinds["regressed"]} regressed, {kinds["improved"]} improved'
    errored = sum(1 for _, change in pairs if change == 'error')
    if errored:
        title += f', {errored} with an error'
    if cmp.outcome_filter:
        return (
            f'{n} pair{"" if n == 1 else "s"}',
            '',
            title + f'. Goal rate hidden: you filtered by {cmp.outcome_filter}.',
        )
    if not scored:
        return 'err', '', title + '. No pair where both runs finished, so no goal rate.'
    delta = (sum(b for _, b in scored) - sum(a for a, _ in scored)) / len(scored)
    pts = round(delta * 100)
    text = '0 pts' if pts == 0 else f'{pts:+d} pts'.replace('-', _MINUS)
    fill = _FLAT_FILL if pts == 0 else _interp_color((delta + 1) / 2, stops=_DIVERGING)
    return text, fill, title


def _axis(labels: Counter[str], selected: list[str], noun: str) -> tuple[list[str], str]:
    """The labels to draw (busiest first, a selected one always kept) and the "+N more" note."""
    ordered = sorted(labels, key=lambda label: (label not in selected, -labels[label], label))
    kept = ordered[:_MAX_AXIS]
    hidden = len(ordered) - len(kept)
    return kept, f'+{hidden} more {noun}{"" if hidden == 1 else "s"}' if hidden else ''


def where_panel(cmp: Comparison, state: TableState | None = None) -> str:
    """Persona x scenario grid of the goal-rate change; each cell filters the page to that pair.

    The axes come from every matched pair so the grid keeps its shape while a
    persona or scenario filter is on; the values come from the pairs that
    survive the filter. Empty when there is one persona and one scenario.
    """
    personas = Counter(p.a.persona for p in cmp.full.matched)
    scenarios = Counter(p.a.scenario for p in cmp.full.matched)
    if len(personas) < 2 and len(scenarios) < 2:
        return ''
    narrowing = cmp.narrowing
    rows, more_rows = _axis(personas, narrowing.get('persona', []), 'persona')
    cols, more_cols = _axis(scenarios, narrowing.get('scenario', []), 'scenario')
    grouped: dict[tuple[str, str], list[tuple[MatchedPair, str]]] = {}
    for pair, change in cmp.changes:
        grouped.setdefault((pair.a.persona, pair.a.scenario), []).append((pair, change))
    cells = [{'persona': p, 'scenario': s, 'pairs': pairs} for (p, s), pairs in grouped.items()]
    base = first_page(state)

    def cell(c: dict) -> str:
        persona, scenario = c['persona'], c['scenario']
        text, fill, title = _cell_value(cmp, c['pairs'])
        on = narrowing.get('persona') == [persona] and narrowing.get('scenario') == [scenario]
        target = {k: v for k, v in narrowing.items() if k not in ('persona', 'scenario')}
        if not on:
            target['persona'], target['scenario'] = [persona], [scenario]
        hint = 'Filter on: select to remove it' if on else 'Select to filter to this cell'
        label = f'{persona} and {scenario}: {text}, {title}. {hint}'
        cls = 'rk-heat-cell cmp-heat-cell' + ('' if fill else ' rk-heat-no-verdict') + (' is-on' if on else '')
        style = f' style="background:{fill};color:var(--ink-900)"' if fill else ''
        current = ' aria-current="true"' if on else ''
        return (
            f'<td class="{cls}"{style} title="{esc(title)}">'
            f'<a class="cmp-heat-link" {swap_link(cmp, target, base)} aria-label="{esc(label)}"{current}>'
            f'{esc(text)}</a></td>'
        )

    notes = ' · '.join(n for n in (more_rows, more_cols) if n)
    foot = f'<p class="cmp-note">{esc(notes)}</p>' if notes else ''
    legend = (
        f'<p class="cmp-note">Pairs per cell. Goal rate hidden: you filtered by {esc(cmp.outcome_filter)}</p>'
        if cmp.outcome_filter
        else f'<div class="cmp-legend"><span><i style="background:{_WORSE}"></i>Worse in B</span>'
        f'<span><i style="background:{_FLAT_FILL};box-shadow:inset 0 0 0 1px var(--border-subtle)"></i>No change</span>'
        f'<span><i style="background:{_BETTER}"></i>Better in B</span></div>'
    )
    table = f'<div class="cmp-heat-scroll">{heatmap(rows, cols, cells, render_cell=cell)}</div>'
    return panel(
        'Where it changed', f'{legend}{table}{foot}', sub='Goal-rate change from A to B per persona and scenario'
    )
