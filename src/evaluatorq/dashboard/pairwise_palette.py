"""Side colours and labels shared by the pairwise report modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluatorq.pairwise_run import PairwiseRun


def pw_side_colour(value: str | None) -> str:
    """CSS colour for a winner or vote value; anything but A, B or tie reads as "none"."""
    if value == 'A':
        return 'var(--pw-a)'
    if value == 'B':
        return 'var(--pw-b)'
    if value == 'tie':
        return 'var(--pw-tie)'
    return 'var(--pw-none)'


def pw_side_label(value: str | None, run: PairwiseRun) -> str:
    """Display name for a winner or vote value, using the run's side labels."""
    if value == 'A':
        return run.label_a
    if value == 'B':
        return run.label_b
    if value in ('tie', 'inconclusive'):
        return value
    return 'abstain'
