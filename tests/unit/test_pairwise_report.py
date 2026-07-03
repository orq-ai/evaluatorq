from __future__ import annotations

import pytest

from evaluatorq.pairwise import PairwiseComparison, PairwiseVote, build_report


def _comparisons() -> list[PairwiseComparison]:
    # Judge x: always decisive. Judge y: one flip. Judge z: one tie, one flip.
    return [
        PairwiseComparison(
            winner='A',
            votes=[
                PairwiseVote(model='x', vote='A', flipped=False),
                PairwiseVote(model='y', vote='A', flipped=False),
                PairwiseVote(model='z', vote='tie', flipped=False),
            ],
        ),
        PairwiseComparison(
            winner='B',
            votes=[
                PairwiseVote(model='x', vote='B', flipped=False),
                PairwiseVote(model='y', vote=None, flipped=True),
                PairwiseVote(model='z', vote=None, flipped=True),
            ],
        ),
        PairwiseComparison(
            winner='tie',
            votes=[
                PairwiseVote(model='x', vote='tie', flipped=False),
                PairwiseVote(model='y', vote='A', flipped=False),
                PairwiseVote(model='z', vote='B', flipped=False),
            ],
        ),
    ]


def test_report_counts_comparisons() -> None:
    """The report records how many comparisons went into it."""
    report = build_report(_comparisons())

    assert report.comparisons == 3


def test_contestant_win_rates_come_from_consensus() -> None:
    """Headline win rates are consensus wins over comparisons decided A or B; tie rate is separate."""
    report = build_report(_comparisons())

    # 3 comparisons: A wins 1, B wins 1, tie 1. Decisive (A or B) = 2.
    assert report.a_win_rate == 0.5
    assert report.b_win_rate == 0.5
    assert report.tie_rate == 1 / 3


def test_per_judge_lean_over_decisive_picks() -> None:
    """A judge's win rate is the share of its A/B picks going to each side; ties/abstains excluded."""
    per_judge = {j.model: j for j in build_report(_comparisons()).per_judge}

    # Judge x picked A once and B once (plus a tie, excluded) -> 50/50.
    assert per_judge['x'].a_rate == 0.5
    assert per_judge['x'].b_rate == 0.5
    # Judge y picked A once, no B pick (other vote was a flip/abstain) -> 100/0.
    assert per_judge['y'].a_rate == 1.0
    assert per_judge['y'].b_rate == 0.0


def test_per_judge_position_bias_is_flips_over_total() -> None:
    """Position bias is the flip rate across all comparisons the judge saw."""
    per_judge = {j.model: j for j in build_report(_comparisons()).per_judge}

    assert per_judge['x'].position_bias == 0.0  # never flipped
    assert per_judge['y'].position_bias == 1 / 3  # flipped 1 of 3
    assert per_judge['z'].position_bias == 1 / 3  # flipped 1 of 3


def test_per_judge_tie_rate() -> None:
    """Tie rate is the share of a judge's comparisons it called a tie."""
    per_judge = {j.model: j for j in build_report(_comparisons()).per_judge}

    assert per_judge['x'].tie_rate == 1 / 3  # one tie of three
    assert per_judge['y'].tie_rate == 0.0


def test_judge_with_no_decisive_picks_has_none_lean() -> None:
    """A judge that never picks a side has undefined (None) lean, not a divide-by-zero."""
    comparisons = [
        PairwiseComparison(
            winner='inconclusive',
            votes=[PairwiseVote(model='w', vote=None, flipped=True)],
        )
    ]

    per_judge = {j.model: j for j in build_report(comparisons).per_judge}

    assert per_judge['w'].a_rate is None
    assert per_judge['w'].b_rate is None


def test_report_mean_agreement() -> None:
    """Mean agreement averages each comparison's modal share over its decisive reconciled votes."""
    report = build_report(_comparisons())

    # comp1 [A,A,tie]->2/3, comp2 [B]->1.0, comp3 [tie,A,B]->1/3; mean = 2/3.
    assert report.mean_agreement == pytest.approx(2 / 3)


def test_mean_agreement_none_when_no_decisive_votes() -> None:
    """With no decisive votes anywhere, agreement is undefined (None), not zero."""
    comparisons = [
        PairwiseComparison(
            winner='inconclusive',
            votes=[PairwiseVote(model='w', vote=None, flipped=True)],
        )
    ]

    assert build_report(comparisons).mean_agreement is None
