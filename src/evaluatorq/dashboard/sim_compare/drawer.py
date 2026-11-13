"""The paired-transcript drawer of the sim run comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc
from evaluatorq.common.reports.sim_detail import first_divergence, flipped_criteria
from evaluatorq.dashboard.sim_compare.fmt import change_pill, delta_tone, run_key, status_badge
from evaluatorq.dashboard.sim_compare.model import MatchedPair, Side, pair_change
from evaluatorq.dashboard.sim_views import (
    render_transcript_fragment,
)

if TYPE_CHECKING:
    from evaluatorq.simulation.contracts import SimulationEntry

_CRITERION_GLYPH = {'pass': '&#x2713;', 'fail': '&#x2717;', 'unknown': '?', 'absent': '&ndash;'}


def _differences(entry_a: SimulationEntry, entry_b: SimulationEntry, side_a: Side, side_b: Side) -> str:
    """What changed for this pair, above the transcripts: verdict, goal score,
    turns, the criteria that flipped and where the transcripts part ways. Judge
    reasons stay in each column, beside the criteria they explain."""

    def ab(va: str, vb: str) -> str:
        return f'<span class="cmp-diff-ab">{va} <span class="cmp-arrow">&rarr;</span> {vb}</span>'

    d_score = entry_b.goal_completion_score - entry_a.goal_completion_score
    rows = [
        ('Outcome', ab(status_badge(side_a.status(entry_a)), status_badge(side_b.status(entry_b)))),
        (
            'Goal score',
            ab(f'{entry_a.goal_completion_score:.2f}', f'{entry_b.goal_completion_score:.2f}')
            + f' <span class="cmp-num-d cmp-num-d--{delta_tone(d_score)}">{d_score:+.2f}</span>',
        ),
        ('Turns', ab(str(entry_a.turn_count), str(entry_b.turn_count))),
    ]
    flipped = flipped_criteria(entry_a, entry_b)
    if flipped:
        items = ''.join(
            f'<li><span class="cmp-diff-crit cmp-diff-crit--{sa}">{_CRITERION_GLYPH.get(sa, "?")}</span>'
            f'<span class="cmp-arrow">&rarr;</span>'
            f'<span class="cmp-diff-crit cmp-diff-crit--{sb}">{_CRITERION_GLYPH.get(sb, "?")}</span>'
            f'<span>{esc(desc)}</span></li>'
            for desc, sa, sb in flipped
        )
        rows.append(('Criteria', f'<ul class="cmp-diff-crits">{items}</ul>'))
    elif entry_a.criteria or entry_b.criteria:
        rows.append(('Criteria', '<span class="cmp-diff-same">Same verdict on every criterion</span>'))
    split = first_divergence(entry_a, entry_b)
    rows.append((
        'Transcript',
        '<span class="cmp-diff-same">Identical in both runs</span>'
        if split is None
        else f'First differs at message {split + 1}, marked below',
    ))
    body = ''.join(f'<dt>{label}</dt><dd>{value}</dd>' for label, value in rows)
    return f'<section class="cmp-diff" aria-label="Differences"><dl class="cmp-diff-list">{body}</dl></section>'


def render_compare_transcript(entry_a: SimulationEntry, entry_b: SimulationEntry, side_a: Side, side_b: Side) -> str:
    """Paired transcript for the drawer: the pair named once, a Differences
    block, then both runs' criteria and transcripts side by side. The columns
    share one scroll container, so they scroll together. The first message
    where the transcripts differ is marked in both columns."""

    def column(which: str, entry: SimulationEntry, side: Side) -> str:
        return (
            '<div class="cmp-side">'
            f'<div class="cmp-side-title">{run_key(which)}<span class="cmp-side-name">{esc(side.name)}</span>'
            f'<span class="cmp-side-id">#{entry.index}</span>{status_badge(side.status(entry))}</div>'
            f'{render_transcript_fragment(entry, summary=False, mark_index=split)}</div>'
        )

    split = first_divergence(entry_a, entry_b)
    change = pair_change(side_a, side_b, MatchedPair(entry_a, entry_b))
    head = (
        '<header class="cmp-pair-head">'
        f'<div class="cmp-pair-title">{esc(entry_a.persona)} <span class="cmp-pair-sep">·</span> {esc(entry_a.scenario)}</div>'
        f'{change_pill(change)}</header>'
    )
    return (
        f'{head}{_differences(entry_a, entry_b, side_a, side_b)}'
        f'<div class="cmp-transcript-grid">{column("A", entry_a, side_a)}{column("B", entry_b, side_b)}</div>'
    )
