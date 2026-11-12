"""The "Where it changed" persona x scenario heatmap of goal-rate change from A to B."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.report_kit import _interp_color, panel
from evaluatorq.dashboard.sim_compare.state import TableState, _base_state, _swap_link

if TYPE_CHECKING:
    from evaluatorq.dashboard.sim_compare.model import Comparison, MatchedPair

# Most rows and columns drawn; the rest collapse into a "+N more" note.
_MAX_AXIS = 12

# Diverging fill over a goal-rate change of -100 to +100 pts: red worse, neutral flat, green better.
# Both ends stay light enough for dark ink.
_DIVERGING = ((0.0, '#e58f89'), (0.5, '#eceff1'), (1.0, '#7fc8a5'))

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
    if not scored:
        return 'err', '', title + '. No pair where both runs finished, so no goal rate.'
    delta = (sum(b for _, b in scored) - sum(a for a, _ in scored)) / len(scored)
    pts = round(delta * 100)
    text = '0 pts' if pts == 0 else f'{pts:+d} pts'.replace('-', _MINUS)
    fill = '#eceff1' if pts == 0 else _interp_color((delta + 1) / 2, stops=_DIVERGING)
    return text, fill, title


def _axis(labels: Counter[str], selected: list[str], noun: str) -> tuple[list[str], str]:
    """The labels to draw (busiest first, a selected one always kept) and the "+N more" note."""
    ordered = sorted(labels, key=lambda label: (label not in selected, -labels[label], label))
    kept = ordered[:_MAX_AXIS]
    hidden = len(ordered) - len(kept)
    return kept, f'+{hidden} more {noun}{"" if hidden == 1 else "s"}' if hidden else ''


def _where_panel(cmp: Comparison, state: TableState | None = None) -> str:
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
    cells: dict[tuple[str, str], list[tuple[MatchedPair, str]]] = {}
    for pair, change in cmp.changes:
        cells.setdefault((pair.a.persona, pair.a.scenario), []).append((pair, change))
    base = _base_state(state)

    def cell(persona: str, scenario: str) -> str:
        pairs = cells.get((persona, scenario))
        if not pairs:
            return '<td class="rk-heat-cell rk-heat-empty">&mdash;</td>'
        text, fill, title = _cell_value(cmp, pairs)
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
            f'<a class="cmp-heat-link" {_swap_link(cmp, target, base)} aria-label="{esc(label)}"{current}>'
            f'{esc(text)}</a></td>'
        )

    header = ''.join(f'<th class="rk-heat-col">{esc(s)}</th>' for s in cols)
    body = ''.join(f'<tr><th class="rk-heat-row">{esc(p)}</th>{"".join(cell(p, s) for s in cols)}</tr>' for p in rows)
    notes = ' · '.join(n for n in (more_rows, more_cols) if n)
    foot = f'<p class="cmp-note">{esc(notes)}</p>' if notes else ''
    legend = (
        '<div class="cmp-legend"><span><i style="background:#e58f89"></i>Worse in B</span>'
        '<span><i style="background:#eceff1;box-shadow:inset 0 0 0 1px var(--border-subtle,#d6dade)"></i>No change</span>'
        '<span><i style="background:#7fc8a5"></i>Better in B</span></div>'
    )
    table = (
        f'<div class="cmp-heat-scroll"><table class="rk-heatmap"><thead><tr><th></th>{header}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )
    return panel(
        'Where it changed', f'{legend}{table}{foot}', sub='Goal-rate change from A to B per persona and scenario'
    )
