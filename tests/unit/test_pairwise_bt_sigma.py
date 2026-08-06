"""BT-sigma aggregation over pairwise jury runs.

Pins the integration contract: the default report is byte-identical to before,
and the opt-in reliability-weighted aggregation down-weights judges the fit
marks as noisy, flipping consensus where a uniform plurality is outvoted by
unreliable judges.
"""

from __future__ import annotations

import pytest

from evaluatorq.pairwise import (
    PairwiseComparison,
    PairwiseVote,
    bt_sigma_aggregation,
    build_report,
    pairwise_consensus,
)


def _vote(model: str, value: str | None) -> PairwiseVote:
    return PairwiseVote(model=model, vote=value)


def _comparison(votes: list[PairwiseVote]) -> PairwiseComparison:
    return PairwiseComparison(winner=pairwise_consensus([v.vote for v in votes]), votes=votes)


def _heterogeneous_run() -> list[PairwiseComparison]:
    """One steady judge always voting A; two noisy judges that flip sides.

    The steady judge is perfectly consistent with a fixed A-beats-B gap, so the
    fit assigns it a small sigma. The noisy pair split across rows, earning
    large sigmas. On rows where both noisy judges vote B, plurality says B but
    the reliability-weighted vote should say A.
    """
    rows = []
    for k in range(12):
        noisy_1 = 'A' if k % 2 == 0 else 'B'
        noisy_2 = 'A' if k % 3 == 0 else 'B'
        rows.append(_comparison([_vote('steady', 'A'), _vote('noisy-1', noisy_1), _vote('noisy-2', noisy_2)]))
    return rows


def test_default_report_is_unchanged() -> None:
    report = build_report(_heterogeneous_run())
    assert report.bt_sigma is None
    assert all(stats.sigma is None for stats in report.per_judge)


def test_unknown_aggregation_raises() -> None:
    with pytest.raises(ValueError, match='unknown aggregation'):
        build_report(_heterogeneous_run(), aggregation='elo')


def test_bt_sigma_downweights_noisy_judges_and_flips_consensus() -> None:
    rows = _heterogeneous_run()
    report = build_report(rows, aggregation='bt-sigma')
    block = report.bt_sigma
    assert block is not None
    assert block.judge_sigmas['steady'] < block.judge_sigmas['noisy-1']
    assert block.judge_sigmas['steady'] < block.judge_sigmas['noisy-2']
    # Rows where the two noisy judges outvote the steady one under plurality:
    outvoted = [i for i, row in enumerate(rows) if row.winner == 'B']
    assert outvoted, 'fixture must contain plurality-B rows'
    # The weighted vote restores A on those rows.
    assert all(block.winners[i] == 'A' for i in outvoted)
    assert block.p_a_beats_b > 0.5
    assert block.skill_gap > 0.0
    # Weighted rollup beats the plurality rollup toward the steady signal.
    weighted_a = block.a_win_rate
    plurality_a = report.a_win_rate
    assert weighted_a == 1.0
    assert weighted_a is not None
    assert plurality_a is not None
    assert weighted_a > plurality_a


def test_judge_stats_carry_sigma_when_requested() -> None:
    report = build_report(_heterogeneous_run(), aggregation='bt-sigma')
    by_model = {stats.model: stats.sigma for stats in report.per_judge}
    steady, noisy = by_model['steady'], by_model['noisy-1']
    assert steady is not None
    assert noisy is not None
    assert steady < noisy


def test_headline_plurality_rates_stay_comparable() -> None:
    rows = _heterogeneous_run()
    plain = build_report(rows)
    with_bt = build_report(rows, aggregation='bt-sigma')
    # The plurality headline is identical either way; BT-sigma is additive.
    assert with_bt.a_win_rate == plain.a_win_rate
    assert with_bt.b_win_rate == plain.b_win_rate
    assert with_bt.inconclusive_rate == plain.inconclusive_rate


def test_abstaining_panel_degrades_to_inconclusive() -> None:
    rows = [_comparison([_vote('a', None), _vote('b', None)]) for _ in range(3)]
    block = bt_sigma_aggregation(rows)
    assert block.winners == ['inconclusive'] * 3
    assert block.a_win_rate is None
    assert block.inconclusive_rate == 1.0
    assert any('no decisive votes' in w for w in block.fit_warnings)


def test_single_judge_run_weights_uniformly() -> None:
    rows = [_comparison([_vote('only', 'A')]) for _ in range(4)]
    block = bt_sigma_aggregation(rows)
    # Sigma is unidentifiable with one judge; the fit falls back and the
    # weighted vote reduces to the plain one.
    assert block.judge_sigmas == {}
    assert block.winners == ['A'] * 4
    assert any('single judge' in w for w in block.fit_warnings)


def test_tie_between_weighted_sides_is_inconclusive() -> None:
    # Two judges, perfectly symmetric opposite votes: sigmas equal, weights
    # equal, every row splits -> inconclusive, matching plurality semantics.
    rows = [_comparison([_vote('a', 'A'), _vote('b', 'B')]) for _ in range(6)]
    block = bt_sigma_aggregation(rows)
    assert set(block.winners) == {'inconclusive'}


def test_run_rollup_and_save_thread_aggregation(tmp_path) -> None:
    from evaluatorq.pairwise_run import PairwiseRun, new_run

    run = new_run(run_name='bt-sigma-check')
    for row in _heterogeneous_run():
        run.add(row, question='q', response_a='a', response_b='b')

    # A plurality-saved report is not silently reused for a bt-sigma request.
    path = run.save(tmp_path / 'run.json')
    assert path.exists()
    assert run.report is not None
    assert run.report.bt_sigma is None
    weighted = run.rollup(aggregation='bt-sigma')
    assert weighted.bt_sigma is not None

    # Saving with bt-sigma persists the block, and rollup then reuses it.
    run.save(tmp_path / 'run2.json', aggregation='bt-sigma')
    assert run.report is not None
    assert run.report.bt_sigma is not None
    assert run.rollup(aggregation='bt-sigma') is run.report
