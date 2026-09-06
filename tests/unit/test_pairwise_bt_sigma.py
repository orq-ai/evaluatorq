"""BT-sigma aggregation over pairwise jury runs.

Pins the integration contract: the default report is byte-identical to before,
and the opt-in reliability-weighted aggregation down-weights judges the fit
marks as noisy, flipping consensus where a uniform plurality is outvoted by
unreliable judges.
"""

from __future__ import annotations

from typing import Literal, cast

import pytest
from pydantic import ValidationError

from evaluatorq.pairwise import (
    PairwiseComparison,
    PairwiseVote,
    bt_sigma_aggregation,
    build_report,
    pairwise_consensus,
)

_Vote = Literal["A", "B", "tie"]


def _vote(model: str, value: str | None) -> PairwiseVote:
    return PairwiseVote(model=model, vote=cast("Literal['A', 'B', 'tie'] | None", value))


def _comparison(votes: list[PairwiseVote]) -> PairwiseComparison:
    return PairwiseComparison(winner=pairwise_consensus([v.vote for v in votes]), votes=votes)


def _heterogeneous_run() -> list[PairwiseComparison]:
    """One steady judge voting A on 11 of 12 rows; two noisy judges that flip.

    The steady judge votes B once (row 1, where the noisy pair also votes B) so
    it stays out of the one-sided guard and its small sigma is a genuine
    reliability estimate. The noisy pair split across rows, earning large
    sigmas. On rows where both noisy judges outvote a steady A under plurality,
    the reliability-weighted vote should restore A.
    """
    rows = []
    for k in range(12):
        steady: _Vote = "B" if k == 1 else "A"
        noisy_1: _Vote = "A" if k % 2 == 0 else "B"
        noisy_2: _Vote = "A" if k % 3 == 0 else "B"
        rows.append(_comparison([_vote("steady", steady), _vote("noisy-1", noisy_1), _vote("noisy-2", noisy_2)]))
    return rows


def test_default_report_is_unchanged() -> None:
    report = build_report(_heterogeneous_run())
    assert report.bt_sigma is None
    assert all(stats.sigma is None for stats in report.per_judge)
    # This run records no repetition observations, so None here is the honest "no usable repeats"
    # and not the old bt-sigma gate. Repetition runs assert the populated case in
    # tests/unit/test_pairwise_repetitions.py.
    assert all(stats.consistency is None and stats.consistency_raw is None for stats in report.per_judge)


def test_unknown_aggregation_raises() -> None:
    with pytest.raises(ValueError, match='unknown aggregation'):
        build_report(_heterogeneous_run(), aggregation='elo')


def test_invalid_vote_is_rejected_at_the_public_model_boundary() -> None:
    with pytest.raises(ValidationError):
        PairwiseVote(model='judge', vote='unexpected')  # pyright: ignore[reportArgumentType]


def test_bt_sigma_downweights_noisy_judges_and_flips_consensus() -> None:
    rows = _heterogeneous_run()
    report = build_report(rows, aggregation='bt-sigma')
    block = report.bt_sigma
    assert block is not None
    assert block.converged is True
    assert block.judge_sigmas['steady'] < block.judge_sigmas['noisy-1']
    assert block.judge_sigmas['steady'] < block.judge_sigmas['noisy-2']
    # Rows where the two noisy judges outvote a steady A under plurality:
    outvoted = [i for i, row in enumerate(rows) if row.winner == 'B' and row.votes[0].vote == 'A']
    assert outvoted, 'fixture must contain plurality-B rows against a steady A'
    # The weighted vote restores A on those rows.
    assert all(block.winners[i] == 'A' for i in outvoted)
    # Row 1 is unanimous B and must stay B under any weighting.
    assert block.winners[1] == 'B'
    assert block.p_a_beats_b > 0.5
    assert block.skill_gap > 0.0
    # Weighted rollup beats the plurality rollup toward the steady signal.
    weighted_a = block.a_win_rate
    plurality_a = report.a_win_rate
    assert weighted_a is not None
    assert plurality_a is not None
    assert weighted_a == pytest.approx(11 / 12)
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


def test_non_converged_fit_does_not_change_plurality_winners(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.ranking as ranking_mod

    monkeypatch.setattr(ranking_mod, '_MAX_ITER', 1)
    rows = [_comparison([_vote('a', 'A'), _vote('b', 'A')]) for _ in range(10)]
    block = bt_sigma_aggregation(rows)
    assert block.judge_sigmas == {}
    assert block.winners == ['A'] * len(rows)
    assert block.converged is False
    assert any('non-converged fit' in warning for warning in block.fit_warnings)


def test_single_ordering_fit_does_not_claim_reliability() -> None:
    rows = [_comparison([_vote('a', 'A'), _vote('b', 'B')]) for _ in range(3)]
    for row in rows:
        for vote in row.votes:
            vote.completed = False
    block = bt_sigma_aggregation(rows)
    assert block.judge_sigmas == {}
    assert block.winners == ['inconclusive'] * len(rows)
    assert any('single-ordering' in warning for warning in block.fit_warnings)


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


def test_one_sided_judge_cannot_take_over_the_run() -> None:
    # The production shape of a position- or verbosity-biased judge: it never
    # picks B at all. In the two-item fit its sigma collapses toward zero
    # (sigma is pinned by the judge's own one-sidedness), so 1/sigma weighting
    # would let it outvote an agreeing two-judge majority on every row. The
    # guard weights it neutrally instead and says so.
    rows = []
    for k in range(10):
        majority: Literal["A", "B"] = "B" if k < 6 else "A"
        rows.append(_comparison([_vote("always-a", "A"), _vote("y", majority), _vote("z", majority)]))
    block = bt_sigma_aggregation(rows)
    assert "always-a" not in block.judge_sigmas
    assert any("always-a" in w and "one-sidedness" in w for w in block.fit_warnings)
    # The agreeing majority keeps every row it won under plurality.
    assert block.winners == [row.winner for row in rows]
    assert block.b_win_rate is not None
    assert block.b_win_rate == pytest.approx(6 / 10)


def test_tie_votes_flow_through_the_aggregation() -> None:
    # End-to-end 'tie' coverage: tie maps to p=0.5 in the fit, and a weighted
    # tie leader stays 'tie' in the winners.
    pattern: list[_Vote] = ["A", "A", "tie", "A", "B", "tie"]
    rows = [_comparison([_vote("a", pattern[k]), _vote("b", pattern[k])]) for k in range(6)]
    block = bt_sigma_aggregation(rows)
    assert block.winners[2] == "tie"
    assert block.winners[5] == "tie"
    assert block.tie_rate == pytest.approx(2 / 6)
    assert block.p_a_beats_b > 0.5


def test_cap_hit_reports_unconverged(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.ranking as ranking_mod

    monkeypatch.setattr(ranking_mod, "_MAX_ITER", 3)
    block = bt_sigma_aggregation(_heterogeneous_run())
    assert block.converged is False
    assert any("iteration cap" in w for w in block.fit_warnings)


def test_low_vote_count_judges_are_flagged() -> None:
    # Three decisive votes per judge, mixed so the one-sided guard stays out of
    # the way: sigmas exist but are mostly prior, and fit_warnings says so.
    pattern_a: list[_Vote] = ["A", "B", "A"]
    pattern_b: list[_Vote] = ["B", "A", "A"]
    rows = [_comparison([_vote("a", pattern_a[k]), _vote("b", pattern_b[k])]) for k in range(3)]
    block = bt_sigma_aggregation(rows)
    assert any("only 3 decisive vote" in w for w in block.fit_warnings)


def test_run_rollup_and_save_thread_aggregation(tmp_path) -> None:
    from evaluatorq.pairwise_run import new_run

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

    with pytest.raises(ValueError, match='unknown aggregation'):
        run.rollup(aggregation='elo')


def test_heterogeneous_entries_warn_about_global_sigma_scope() -> None:
    from evaluatorq.pairwise_run import new_run

    run = new_run(run_name='heterogeneous')
    for index, row in enumerate(_heterogeneous_run()[:2]):
        run.add(row, question=f'q-{index}', response_a=f'a-{index}', response_b=f'b-{index}')

    report = run.rollup(aggregation='bt-sigma')
    assert report.bt_sigma is not None
    assert any('global agreement' in warning for warning in report.bt_sigma.fit_warnings)


def test_report_sections_expose_bt_sigma_and_scope_warning(tmp_path) -> None:
    from evaluatorq.pairwise_reports.sections import build_report_sections
    from evaluatorq.pairwise_run import new_run

    run = new_run(run_name='sections', judges=['steady', 'noisy'])
    for index, row in enumerate(_heterogeneous_run()[:4]):
        run.add(row, question=f'q-{index}', response_a=f'a-{index}', response_b=f'b-{index}')
    run.save(tmp_path / 'sections.json', aggregation='bt-sigma')

    consensus = build_report_sections(run)[0]
    judges = build_report_sections(run)[1]
    assert consensus.data['bt_sigma'] is not None
    assert any('global agreement' in warning for warning in consensus.data['bt_sigma']['fit_warnings'])
    assert all('sigma' in row for row in judges.data['rows'])
