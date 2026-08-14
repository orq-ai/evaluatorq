"""Repetition-aware BT-sigma reliability (RES-1251).

Pins the contract: per-repetition votes are preserved and canonicalized on
PairwiseVote, reliability weights come from within-datapoint repetition
consistency when repeats exist (heterogeneous datapoints structurally cannot
distort them), repetition count never multiplies a judge's panel weight, and
legacy runs without observations keep the previous global-fit behaviour.
"""

from __future__ import annotations

import asyncio
from typing import Literal, cast

from evaluatorq.common.jury import Prediction
from evaluatorq.pairwise import (
    PairwiseComparison,
    PairwiseVote,
    RepetitionObservation,
    bt_sigma_aggregation,
    build_report,
    pairwise_consensus,
    repetition_consistency,
    run_pairwise,
)

_Vote = Literal['A', 'B', 'tie']


def _obs(ordering: str, repetition: int, verdict: str | None) -> RepetitionObservation:
    return RepetitionObservation(
        ordering=cast("Literal['ab', 'ba']", ordering),
        repetition=repetition,
        verdict=cast("Literal['A', 'B', 'tie'] | None", verdict),
    )


def _vote(
    model: str,
    value: str | None,
    observations: list[RepetitionObservation] | None = None,
    repetition_failures: int = 0,
) -> PairwiseVote:
    return PairwiseVote(
        model=model,
        vote=cast("Literal['A', 'B', 'tie'] | None", value),
        observations=observations or [],
        repetition_failures=repetition_failures,
    )


def _comparison(votes: list[PairwiseVote]) -> PairwiseComparison:
    return PairwiseComparison(winner=pairwise_consensus([v.vote for v in votes]), votes=votes)


# ---------------------------------------------------------------------------
# Capture + canonicalization (through the real run_pairwise pipeline)
# ---------------------------------------------------------------------------


def test_run_pairwise_captures_and_canonicalizes_repetitions() -> None:
    """A slot-A-biased judge answers 'A' in both orderings; the swapped
    ordering's entries must be recorded as 'B' (original orientation)."""

    async def judge(a: object, b: object, model: str) -> Prediction:
        return Prediction(value='A', explanation='first slot looks great')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['biased'], response_a='x', response_b='y', repetitions=2)
    )
    (vote,) = comparison.votes
    ab = [o for o in vote.observations if o.ordering == 'ab']
    ba = [o for o in vote.observations if o.ordering == 'ba']
    assert [o.verdict for o in ab] == ['A', 'A']
    assert [o.verdict for o in ba] == ['B', 'B']
    assert [o.repetition for o in ab] == [0, 1]
    # Reconciliation still abstains the position-biased judge; the raw
    # observations are preserved regardless (not silently collapsed).
    assert vote.vote is None


def test_failed_and_abstained_repetitions_are_recorded_as_none() -> None:
    calls = {'n': 0}

    async def judge(a: object, b: object, model: str) -> Prediction:
        calls['n'] += 1
        if calls['n'] % 2 == 0:
            raise RuntimeError('boom')
        return Prediction(value='A', explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['flaky'], response_a='x', response_b='y', repetitions=2, swap=False)
    )
    (vote,) = comparison.votes
    assert [o.verdict for o in vote.observations] == ['A', None]
    assert vote.repetition_failures == 1


def test_off_contract_repetition_counts_as_a_failure_not_a_silent_none() -> None:
    # A decisive aggregate ('A') with one off-contract pass ('banana'): the bad
    # pass records as None AND is counted, so it lowers reliability rather than
    # silently passing as a free abstention (RES-1251 review).
    calls = {'n': 0}

    async def judge(a: object, b: object, model: str) -> Prediction:
        calls['n'] += 1
        return Prediction(value=['A', 'banana', 'A'][(calls['n'] - 1) % 3], explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['weird'], response_a='x', response_b='y', repetitions=3, swap=False)
    )
    (vote,) = comparison.votes
    assert vote.vote == 'A'
    assert [o.verdict for o in vote.observations] == ['A', None, 'A']
    assert vote.repetition_failures == 1


# ---------------------------------------------------------------------------
# Consistency semantics
# ---------------------------------------------------------------------------


def test_consistency_groups_are_per_datapoint_and_per_ordering() -> None:
    """Position bias must not read as inconsistency: all-'A' in ab and
    all-'B' (canonicalized) in ba is perfectly consistent within each group."""
    votes = [_vote('biased', None, [_obs('ab', 0, 'A'), _obs('ab', 1, 'A'), _obs('ba', 0, 'B'), _obs('ba', 1, 'B')])]
    consistency = repetition_consistency([_comparison(votes)])
    assert consistency == {'biased': 1.0}


def test_heterogeneous_datapoints_cannot_distort_consistency() -> None:
    """A judge that is perfectly self-consistent on each datapoint but votes
    differently ACROSS datapoints stays at consistency 1.0: different
    questions are never treated as interchangeable observations."""
    c1 = _comparison([_vote('j', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A')])])
    c2 = _comparison([_vote('j', 'B', [_obs('ab', 0, 'B'), _obs('ab', 1, 'B')])])
    assert repetition_consistency([c1, c2]) == {'j': 1.0}


def test_single_repetition_yields_no_consistency_evidence() -> None:
    c = _comparison([_vote('j', 'A', [_obs('ab', 0, 'A'), _obs('ba', 0, 'A')])])
    assert repetition_consistency([c]) == {}


# ---------------------------------------------------------------------------
# Aggregation behaviour
# ---------------------------------------------------------------------------


def _run_with_repeats() -> list[PairwiseComparison]:
    """12 rows; 'steady' self-consistent (weight ~1), 'coin' self-inconsistent
    (weight floored). Steady votes B once so the one-sided guard stays out of
    the way of the underlying fit."""
    rows = []
    for k in range(12):
        steady: _Vote = 'B' if k == 1 else 'A'
        coin: _Vote = 'B' if k != 1 else 'A'  # always outvotes steady 1-1; ties break on weight
        rows.append(
            _comparison([
                _vote('steady', steady, [_obs('ab', 0, steady), _obs('ab', 1, steady)]),
                _vote('coin', coin, [_obs('ab', 0, 'A'), _obs('ab', 1, 'B')]),
            ])
        )
    return rows


def test_weights_come_from_consistency_and_decide_winners() -> None:
    block = bt_sigma_aggregation(_run_with_repeats())
    rc = block.repetition_consistency
    # Shrunk toward the panel mean (RES-1251), so strictly inside (0, 1), but the
    # self-consistent judge is still clearly separated from the coin-flipper.
    assert 0.9 < rc['steady'] < 1.0
    assert 0.0 < rc['coin'] < 0.1
    # The self-consistent judge wins every row against the coin-flipper.
    assert block.winners == ['A' if k != 1 else 'B' for k in range(12)]
    assert any('repetition consistency' in w for w in block.fit_warnings)


def test_judge_stats_surface_repetition_consistency() -> None:
    # The judges table must show the reliability that decided the winners, not
    # only the pooled sigma (RES-1251 review). build_report carries consistency
    # onto each JudgeStats.
    report = build_report(_run_with_repeats(), aggregation='bt-sigma')
    by_model = {j.model: j for j in report.per_judge}
    assert by_model['steady'].consistency is not None and by_model['steady'].consistency > 0.9
    assert by_model['coin'].consistency is not None and by_model['coin'].consistency < 0.1


def test_repetition_count_does_not_multiply_panel_weight() -> None:
    """A judge with 5 repetitions per row must not outvote two judges with 2:
    each judge still casts exactly one weighted vote per datapoint."""
    rows = []
    for _ in range(6):
        many = _vote('many-reps', 'A', [_obs('ab', i, 'A') for i in range(5)])
        few_1 = _vote('few-1', 'B', [_obs('ab', 0, 'B'), _obs('ab', 1, 'B')])
        few_2 = _vote('few-2', 'B', [_obs('ab', 0, 'B'), _obs('ab', 1, 'B')])
        rows.append(_comparison([many, few_1, few_2]))
    block = bt_sigma_aggregation(rows)
    # All three judges are perfectly self-consistent (equal weight); B wins 2 votes to 1.
    assert set(block.winners) == {'B'}


def test_judge_without_repeats_gets_fallback_weight_and_an_honest_warning() -> None:
    rows = _run_with_repeats()
    for row in rows:
        row.votes.append(_vote('legacy-judge', 'A'))
    block = bt_sigma_aggregation(rows)
    assert 'legacy-judge' not in block.repetition_consistency
    # The warning names the fallback (median measured consistency), not "neutral".
    assert any(
        'legacy-judge' in w and 'median measured consistency' in w and 'neutrally' not in w for w in block.fit_warnings
    )


def test_failed_pass_lowers_consistency_but_a_clean_abstention_does_not() -> None:
    # Two judges each agree on their two decisive passes, but 'flaky' errored on a
    # third pass (repetition_failures=1) while 'careful' cleanly abstained on its
    # third. The failure discounts reliability; the abstention does not.
    flaky = _vote('flaky', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A'), _obs('ab', 2, None)], repetition_failures=1)
    careful = _vote('careful', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A'), _obs('ab', 2, None)])
    consistency = repetition_consistency([_comparison([flaky, careful])])
    # The errored pass discounts flaky's reliability (agreement 1.0 x completion 2/3 before
    # shrinkage); the clean abstention costs careful nothing, so careful outranks flaky.
    assert consistency['careful'] > consistency['flaky']


def test_one_sided_warning_on_repetition_path_matches_actual_weighting() -> None:
    """When repetition consistency drives the weights, the one-sided guard
    must not claim the judge was 'weighted neutrally': the neutral assignment
    is replaced by the consistency weights. Only the sigma exclusion holds."""
    rows = _run_with_repeats()
    for row in rows:
        row.votes.append(_vote('always-a', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A')]))
    block = bt_sigma_aggregation(rows)
    warning = next(w for w in block.fit_warnings if 'always-a' in w and 'one-sidedness' in w)
    assert 'excluded from judge_sigmas' in warning
    assert 'weight comes from repetition consistency' in warning
    assert 'weighted neutrally' not in warning
    assert 'always-a' not in block.judge_sigmas
    # And the claim is true: the judge carries its own consistency weight.
    assert block.repetition_consistency['always-a'] > 0.9


def test_reliability_weighting_needs_a_quorum_of_measured_judges() -> None:
    # Only one judge has repetition evidence, so the panel must not be weighted off
    # that single judge's value; repetition weighting is skipped for the pooled fit.
    agree = [_obs('ab', 0, 'A'), _obs('ab', 1, 'A')]
    rows = [_comparison([_vote('measured', 'A', agree), _vote('bare', 'B', [])]) for _ in range(6)]
    block = bt_sigma_aggregation(rows)
    assert any('repetition weighting skipped' in w and 'need >= 2' in w for w in block.fit_warnings)


def test_shrinkage_damps_a_judge_with_thin_evidence() -> None:
    # 'many' and 'few' both agree on every pass (raw 1.0), but 'many' has many
    # comparisons of evidence and 'few' has one. A 'noisy' judge pulls the panel
    # mean below 1.0, so the thin-evidence judge is shrunk further toward it.
    agree = [_obs('ab', 0, 'A'), _obs('ab', 1, 'A')]
    disagree = [_obs('ab', 0, 'A'), _obs('ab', 1, 'B')]
    rows = [_comparison([_vote('many', 'A', agree), _vote('noisy', None, disagree)]) for _ in range(8)]
    rows.append(_comparison([_vote('few', 'A', agree), _vote('noisy', None, disagree)]))
    consistency = repetition_consistency(rows)
    assert consistency['many'] > consistency['few']  # same raw 1.0, more evidence -> less shrinkage
    assert consistency['few'] < 1.0  # thin evidence pulled toward the panel mean


def test_repeat_datapoint_count_requires_repeats_within_one_ordering() -> None:
    """ab=['A', None] plus ba=['A', None] has two decisive observations in
    total but no ordering group with two, so it yields zero consistency
    evidence and must not be counted as a datapoint with repeats."""
    rows = _run_with_repeats()  # 12 rows with genuine within-ordering repeats
    rows.append(
        _comparison([
            _vote(
                'steady',
                'A',
                [_obs('ab', 0, 'A'), _obs('ab', 1, None), _obs('ba', 0, 'A'), _obs('ba', 1, None)],
            )
        ])
    )
    block = bt_sigma_aggregation(rows)
    warning = next(w for w in block.fit_warnings if 'datapoint(s) with repeats' in w)
    assert '12 datapoint(s) with repeats' in warning


def test_legacy_runs_without_observations_keep_global_fit_behaviour() -> None:
    """Old saved runs (no observations anywhere) must aggregate exactly as
    before: no consistency block, no repetition warnings."""
    rows = [
        _comparison([_vote('steady', 'B' if k == 1 else 'A'), _vote('noisy', 'A' if k % 2 == 0 else 'B')])
        for k in range(12)
    ]
    block = bt_sigma_aggregation(rows)
    assert block.repetition_consistency == {}
    assert not any('repetition' in w for w in block.fit_warnings)


def test_legacy_serialized_vote_deserializes_with_empty_observations() -> None:
    vote = PairwiseVote.model_validate({'model': 'j', 'vote': 'A'})
    assert vote.observations == []
    assert vote.repetition_failures == 0


def test_nonconverged_fit_still_uses_consistency_weights(monkeypatch) -> None:
    """The repetition path must not depend on the pooled fit's health: a
    non-converged fit loses only the run-level headline."""
    import evaluatorq.ranking as ranking

    real_fit = ranking.fit_bt

    def broken_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        try:
            fit.converged = False
        except (AttributeError, TypeError, ValueError):  # frozen model
            fit = fit.model_copy(update={'converged': False})
        return fit

    monkeypatch.setattr(ranking, 'fit_bt', broken_fit)
    block = bt_sigma_aggregation(_run_with_repeats())
    assert block.converged is False
    assert block.p_a_beats_b == 0.5
    assert block.skill_gap == 0.0
    assert 0.9 < block.repetition_consistency['steady'] < 1.0
    assert 0.0 < block.repetition_consistency['coin'] < 0.1
    # Winners still come from consistency weights, not uniform plurality.
    assert block.winners == ['A' if k != 1 else 'B' for k in range(12)]
    assert any('non-converged' in w for w in block.fit_warnings)


def test_comparison_with_observations_round_trips_through_json() -> None:
    rows = _run_with_repeats()
    dumped = rows[0].model_dump_json()
    restored = PairwiseComparison.model_validate_json(dumped)
    assert restored == rows[0]
    assert restored.votes[0].observations[0].ordering == 'ab'


def test_end_to_end_pipeline_with_repetitions() -> None:
    """Full path: run_pairwise (repetitions=3, both orderings) over several
    comparisons, then bt_sigma_aggregation. A deterministic judge and a
    call-order coin-flipper; the deterministic judge must carry the vote."""
    flip = {'n': 0}

    async def judge(a: object, b: object, model: str) -> Prediction:
        if model == 'steady':
            return Prediction(value='A' if str(a) == 'good' else 'B', explanation='ok')
        flip['n'] += 1
        return Prediction(value='A' if flip['n'] % 2 else 'B', explanation='hmm')

    async def run_rows() -> list[PairwiseComparison]:
        return [
            await run_pairwise(
                judge_fn=judge, panel=['steady', 'coin'], response_a='good', response_b='bad', repetitions=3
            )
            for _ in range(4)
        ]

    rows = asyncio.run(run_rows())
    for row in rows:
        steady_vote = next(v for v in row.votes if v.model == 'steady')
        assert len(steady_vote.observations) == 6  # 3 reps x 2 orderings
        assert {o.verdict for o in steady_vote.observations} == {'A'}  # canonicalized both orderings
    block = bt_sigma_aggregation(rows)
    assert block.repetition_consistency['steady'] > block.repetition_consistency['coin']
    assert block.repetition_consistency['coin'] < 1.0
    assert set(block.winners) == {'A'}


def test_cost_model_calls_scale_linearly_with_repetitions() -> None:
    """The feasibility claim on the ticket: judge calls = judges x orderings x R."""
    for reps, expected in ((1, 2), (2, 4), (3, 6)):
        calls = {'n': 0}

        async def judge(a: object, b: object, model: str) -> Prediction:
            calls['n'] += 1
            return Prediction(value='A', explanation='ok')

        asyncio.run(run_pairwise(judge_fn=judge, panel=['j'], response_a='x', response_b='y', repetitions=reps))
        assert calls['n'] == expected
