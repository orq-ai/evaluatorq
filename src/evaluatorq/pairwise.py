"""Pairwise (preference) judging over the shared jury.

A pairwise comparison asks judges to pick between two responses, A and B. To
control for position bias each judge is run in both orderings; this module holds
the ordering-independent core: reconciling a judge's two verdicts into one
consistency-gated vote (RES-760, ADR-24).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.jury import VerdictValue, _plurality_vote

if TYPE_CHECKING:
    from collections.abc import Sequence


def reconcile_pair(first: VerdictValue | None, second: VerdictValue | None) -> tuple[VerdictValue | None, bool]:
    """Reconcile a judge's verdicts from both orderings into one vote.

    Both verdicts are in the canonical A/B frame (the caller un-swaps the
    second ordering). Returns ``(vote, flipped)``: a consistent judge votes for
    the value it gave both times; an inconsistent one abstains (``vote=None``)
    and is recorded as a flip. A missing verdict abstains without a flip.
    """
    if first is None or second is None:
        return None, False
    if first == second:
        return first, False
    return None, True


def pairwise_consensus(votes: Sequence[VerdictValue | None]) -> str:
    """Reduce reconciled per-judge votes to a consensus winner.

    Abstained judges (``None``) are dropped. Returns the strict plurality of the
    remaining votes (``'A'``, ``'B'``, or ``'tie'``), or ``'inconclusive'`` when
    no judge had a preference or the vote splits with no plurality.
    """
    decisive = [v for v in votes if v is not None]
    winner, _tie = _plurality_vote(decisive)
    if winner is None:
        return 'inconclusive'
    return str(winner)
