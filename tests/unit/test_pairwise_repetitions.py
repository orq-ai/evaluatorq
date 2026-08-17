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
    repetition_consistency_raw,
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
    # The fallback WEIGHT is actually applied, not just warned about (RES-1251 review,
    # item 10). On row 1 steady (high consistency) votes B while coin and legacy both vote
    # A: uniform plurality would give A (2-1), but the median fallback keeps legacy below
    # steady, so steady's weight still wins the row. Only this row separates the two.
    assert block.winners[1] == 'B'
    assert block.winners == ['A' if k != 1 else 'B' for k in range(12)]


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


def test_failed_pass_penalty_decides_the_winner_through_aggregation() -> None:
    # The failure penalty must change the WINNER, not just the raw consistency number, and
    # it must do so through bt_sigma_aggregation (RES-1251 review, item 10). 'careful' cleanly
    # abstains on its third pass; 'flaky' errors on its third (repetition_failures=1). Both are
    # otherwise self-consistent, so careful's higher reliability decides every row.
    rows = []
    for _ in range(6):
        careful = _vote('careful', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A'), _obs('ab', 2, None)])
        flaky = _vote(
            'flaky', 'B', [_obs('ab', 0, 'B'), _obs('ab', 1, 'B'), _obs('ab', 2, None)], repetition_failures=1
        )
        rows.append(_comparison([careful, flaky]))
    block = bt_sigma_aggregation(rows)
    assert block.repetition_consistency['careful'] > block.repetition_consistency['flaky']
    assert set(block.winners) == {'A'}  # careful's un-penalised reliability wins every row


def test_swapped_ordering_failure_is_counted() -> None:
    # Exercise the SWAPPED ordering's failure path (RES-1251 review, item 11): a judge that
    # answers on the 'ab' ordering (a='x') but errors on the swapped 'ba' ordering (a='y').
    # The ba failures must be counted, not only the swap=False path the other tests cover.
    async def judge(a: object, b: object, model: str) -> Prediction:
        if str(a) == 'y':  # swapped orientation
            msg = 'boom on ba'
            raise RuntimeError(msg)
        return Prediction(value='A', explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['j'], response_a='x', response_b='y', repetitions=2, swap=True)
    )
    (vote,) = comparison.votes
    ba = [o for o in vote.observations if o.ordering == 'ba']
    assert [o.verdict for o in ba] == [None, None]
    assert vote.repetition_failures == 2


def test_single_repetition_through_pipeline_has_no_consistency() -> None:
    # R=1 has no within-ordering repeats, so the REAL pipeline must yield no consistency
    # evidence and fall back cleanly (RES-1251 review, item 11 - the cost test only counted calls).
    async def judge(a: object, b: object, model: str) -> Prediction:
        return Prediction(value='A', explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['a', 'b'], response_a='x', response_b='y', repetitions=1)
    )
    assert repetition_consistency([comparison]) == {}
    block = bt_sigma_aggregation([comparison])
    assert block.repetition_consistency == {}
    assert block.repetition_consistency_raw == {}
    # And R=1 must NOT emit the item-8 "repeated passes exist" warning: no ordering held a genuine
    # repeat, so nothing was skipped (RES-1251 review, item 15 - the warning tripped on the default
    # path when it was gated on observations existing rather than on an actual repeat).
    assert not any('repeated passes exist' in w for w in block.fit_warnings)


def test_html_judges_table_shows_shrunk_and_raw_consistency(tmp_path) -> None:
    # The new columns must render AND map to the right values (RES-1251 review, items 11 + 22):
    # assert each value under its own header's column index, so swapping the two cells fails.
    import re

    from evaluatorq.pairwise_reports.export_html import _render_judges_html
    from evaluatorq.pairwise_reports.sections import build_report_sections
    from evaluatorq.pairwise_run import new_run

    run = new_run(run_name='reps', judges=['steady', 'coin'])
    for i, row in enumerate(_run_with_repeats()):
        run.add(row, question=f'q-{i}', response_a=f'a-{i}', response_b=f'b-{i}')
    run.save(tmp_path / 'run.json', aggregation='bt-sigma')

    judges = build_report_sections(run)[1]
    steady = next(r for r in judges.data['rows'] if r['model'] == 'steady')
    assert steady['consistency_raw'] == 1.0
    assert steady['consistency'] < 1.0  # shrunk below the raw value

    html = _render_judges_html(judges)
    headers = re.findall(r'<th[^>]*>(.*?)</th>', html, re.S)
    assert 'Consistency (shrunk)' in headers
    assert 'Consistency (raw)' in headers
    steady_row = next(r for r in re.findall(r'<tr>(.*?)</tr>', html, re.S) if 'steady' in r)
    cells = re.findall(r'<td[^>]*>(.*?)</td>', steady_row, re.S)
    raw_cell = cells[headers.index('Consistency (raw)')]
    shrunk_cell = cells[headers.index('Consistency (shrunk)')]
    assert '1.00' in raw_cell  # raw self-agreement is under the raw header
    assert '1.00' not in shrunk_cell  # the shrunk weight (< 1.0) is under the shrunk header


def test_repetition_consistency_raw_is_published_alongside_shrunk() -> None:
    # The raw self-agreement is published next to the shrunk weight (RES-1251 review, item 12),
    # so a reader sees the un-shrunk number, not only the reliability weight derived from it.
    block = bt_sigma_aggregation(_run_with_repeats())
    assert block.repetition_consistency_raw['steady'] == 1.0
    assert block.repetition_consistency_raw['coin'] == 0.0
    assert block.repetition_consistency['steady'] < block.repetition_consistency_raw['steady']


def test_observations_without_a_repeated_group_warn() -> None:
    # Observations exist but no ordering group holds >= 2 decisive passes (ab=[A,None],
    # ba=[A,None] at R=2), so rep_consistency comes back EMPTY. The run paid for repeats and
    # fell back to the pooled fit; it must say so rather than stay silent (RES-1251 review, item 8).
    rows = [
        _comparison([
            _vote('j', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, None), _obs('ba', 0, 'A'), _obs('ba', 1, None)]),
            _vote('k', 'B', [_obs('ab', 0, 'B'), _obs('ab', 1, None), _obs('ba', 0, 'B'), _obs('ba', 1, None)]),
        ])
        for _ in range(4)
    ]
    block = bt_sigma_aggregation(rows)
    assert block.repetition_consistency == {}
    assert any('repeated passes exist but no single ordering held' in w for w in block.fit_warnings)


def test_evidence_weighted_shrinkage_anchor_resists_a_thin_noisy_judge() -> None:
    # 'strong' agrees with itself across many comparisons (raw 1.0, high evidence); 'thin'
    # disagrees with itself on its single comparison (raw 0.0). The shrinkage anchor is the
    # EVIDENCE-WEIGHTED panel mean, so the thin judge cannot drag the anchor down and over-shrink
    # strong (RES-1251 review, item 13): a plain mean-of-means anchor (0.5) would pull strong to
    # ~0.95, the evidence-weighted anchor (~0.9) keeps it ~0.99.
    agree = [_obs('ab', 0, 'A'), _obs('ab', 1, 'A')]
    disagree = [_obs('ab', 0, 'A'), _obs('ab', 1, 'B')]
    rows = [_comparison([_vote('strong', 'A', agree)]) for _ in range(8)]
    rows.append(_comparison([_vote('strong', 'A', agree), _vote('thin', None, disagree)]))
    consistency = repetition_consistency(rows)
    assert consistency['strong'] > 0.97
    assert consistency['thin'] < 0.5


def test_valueless_nonabstained_pass_counts_as_a_failure() -> None:
    # A pass with value=None and abstained=False is neither decisive nor a clean abstention: a
    # mechanically-unusable pass. It must count as a repetition failure via the shared jury layer,
    # not slip through as a free abstention keeping consistency at 1.0 (RES-1251 review, item 6).
    calls = {'n': 0}

    async def judge(a: object, b: object, model: str) -> Prediction:
        calls['n'] += 1
        if calls['n'] == 2:
            return Prediction(value=None, abstained=False)
        return Prediction(value='A', explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['j'], response_a='x', response_b='y', repetitions=2, swap=False)
    )
    (vote,) = comparison.votes
    assert vote.repetition_failures == 1


def test_off_contract_repetition_logs_a_warning() -> None:
    # The off-contract collapse must announce itself so a user can find why consistency dropped
    # (RES-1251 review, item 7).
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level='WARNING')
    try:
        calls = {'n': 0}

        async def judge(a: object, b: object, model: str) -> Prediction:
            calls['n'] += 1
            return Prediction(value=['A', 'banana', 'A'][(calls['n'] - 1) % 3], explanation='ok')

        asyncio.run(
            run_pairwise(judge_fn=judge, panel=['weird'], response_a='x', response_b='y', repetitions=3, swap=False)
        )
    finally:
        logger.remove(sink)
    assert any('off-contract' in m for m in messages)


def test_repetition_consistency_raw_is_exported_at_top_level() -> None:
    # The public helper must be importable from the package root and in __all__, like its sibling,
    # so the generated API reference lists it (RES-1251 review, item 16).
    import evaluatorq

    assert hasattr(evaluatorq, 'repetition_consistency_raw')
    assert 'repetition_consistency_raw' in evaluatorq.__all__
    from evaluatorq import repetition_consistency_raw as _rcr  # must not raise

    assert callable(_rcr)


def test_raw_self_agreement_is_not_failure_adjusted() -> None:
    # A judge that agreed with itself on every completed pass reads RAW 1.0 even with a failed pass;
    # only the reliability WEIGHT is discounted by the failure (RES-1251 review, item 17). Previously
    # _repetition_stats applied the discount before returning, so the "raw" field was failure-adjusted.
    flaky = _vote('flaky', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A'), _obs('ab', 2, None)], repetition_failures=1)
    clean = _vote('clean', 'A', [_obs('ab', 0, 'A'), _obs('ab', 1, 'A')])
    rows = [_comparison([flaky, clean]) for _ in range(4)]
    raw = repetition_consistency_raw(rows)
    consistency = repetition_consistency(rows)
    assert raw['flaky'] == 1.0  # agreed with itself on every completed pass; RAW is not discounted
    assert consistency['flaky'] < consistency['clean']  # the failure shows up only in the weight


def test_raw_diagnostic_is_populated_on_the_uniform_fallback_path() -> None:
    # repetition_consistency_raw is a diagnostic, not a weight, so it is published even when weighting
    # is off (RES-1251 review, item 19): a swap=False repetition run still collected the observations.
    async def judge(a: object, b: object, model: str) -> Prediction:
        return Prediction(value='A', explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['j'], response_a='x', response_b='y', repetitions=2, swap=False)
    )
    block = bt_sigma_aggregation([comparison] * 4)
    assert block.repetition_consistency == {}  # weighting off on the single-ordering path
    assert block.repetition_consistency_raw.get('j') == 1.0  # but the raw diagnostic survives


def test_swap_false_repetitions_bypasses_consistency_weighting() -> None:
    # swap=False is a DOCUMENTED gate: even at repetitions>=2 the single-ordering path takes uniform
    # plurality before consistency runs, so removing that guard must fail a test (RES-1251 review, item 22).
    async def judge(a: object, b: object, model: str) -> Prediction:
        return Prediction(value='A', explanation='ok')

    comparison = asyncio.run(
        run_pairwise(judge_fn=judge, panel=['a', 'b'], response_a='x', response_b='y', repetitions=2, swap=False)
    )
    block = bt_sigma_aggregation([comparison] * 4)
    assert block.repetition_consistency == {}
    assert any('single-ordering' in w for w in block.fit_warnings)
