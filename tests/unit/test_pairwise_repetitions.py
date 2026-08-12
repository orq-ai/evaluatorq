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
    pairwise_consensus,
    repetition_consistency,
    run_pairwise,
)

_Vote = Literal["A", "B", "tie"]


def _obs(ordering: str, repetition: int, verdict: str | None) -> RepetitionObservation:
    return RepetitionObservation(
        ordering=cast("Literal['ab', 'ba']", ordering),
        repetition=repetition,
        verdict=cast("Literal['A', 'B', 'tie'] | None", verdict),
    )


def _vote(model: str, value: str | None, observations: list[RepetitionObservation] | None = None) -> PairwiseVote:
    return PairwiseVote(
        model=model,
        vote=cast("Literal['A', 'B', 'tie'] | None", value),
        observations=observations or [],
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


# ---------------------------------------------------------------------------
# Consistency semantics
# ---------------------------------------------------------------------------


def test_consistency_groups_are_per_datapoint_and_per_ordering() -> None:
    """Position bias must not read as inconsistency: all-'A' in ab and
    all-'B' (canonicalized) in ba is perfectly consistent within each group."""
    votes = [
        _vote('biased', None, [_obs('ab', 0, 'A'), _obs('ab', 1, 'A'), _obs('ba', 0, 'B'), _obs('ba', 1, 'B')])
    ]
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
    assert block.repetition_consistency == {'coin': 0.0, 'steady': 1.0}
    # The self-consistent judge wins every row against the coin-flipper.
    assert block.winners == ['A' if k != 1 else 'B' for k in range(12)]
    assert any('repetition consistency' in w for w in block.fit_warnings)


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


def test_judge_without_repeats_gets_neutral_weight_and_a_warning() -> None:
    rows = _run_with_repeats()
    for row in rows:
        row.votes.append(_vote('legacy-judge', 'A'))
    block = bt_sigma_aggregation(rows)
    assert 'legacy-judge' not in block.repetition_consistency
    assert any('legacy-judge' in w and 'neutrally' in w for w in block.fit_warnings)


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
