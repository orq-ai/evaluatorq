from __future__ import annotations

import pytest

from evaluatorq.common.jury import Prediction
from evaluatorq.pairwise import run_pairwise


def _favorite_judge(favorite: str):
    """A consistent judge that always prefers whichever response equals `favorite`."""

    async def judge(first: str, second: str, model: str) -> Prediction:
        if first == favorite:
            return Prediction(value='A', explanation='first is best')
        if second == favorite:
            return Prediction(value='B', explanation='second is best')
        return Prediction(value='tie', explanation='neither')

    return judge


def _first_slot_judge():
    """A position-biased judge that always picks whatever is shown first."""

    async def judge(first: str, second: str, model: str) -> Prediction:
        return Prediction(value='A', explanation='I like the first one')

    return judge


@pytest.mark.asyncio
async def test_consistent_judge_wins_for_the_better_response() -> None:
    """A judge preferring the same actual response in both orderings votes for it, no flip."""
    comparison = await run_pairwise(
        judge_fn=_favorite_judge('GOOD'),
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.winner == 'A'
    assert comparison.votes[0].vote == 'A'
    assert comparison.votes[0].flipped is False


@pytest.mark.asyncio
async def test_position_biased_judge_flips_and_abstains() -> None:
    """A judge that always picks the first slot disagrees with itself once swapped, so it abstains."""
    comparison = await run_pairwise(
        judge_fn=_first_slot_judge(),
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.votes[0].flipped is True
    assert comparison.votes[0].vote is None
    assert comparison.winner == 'inconclusive'


@pytest.mark.asyncio
async def test_swap_off_uses_single_ordering() -> None:
    """With swap disabled the judge runs once and its raw pick stands, never flipped."""
    calls = 0

    async def judge(first: str, second: str, model: str) -> Prediction:
        nonlocal calls
        calls += 1
        return Prediction(value='A', explanation='first')

    comparison = await run_pairwise(
        judge_fn=judge,
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
        swap=False,
    )

    assert calls == 1
    assert comparison.votes[0].vote == 'A'
    assert comparison.votes[0].flipped is False
    assert comparison.winner == 'A'


@pytest.mark.asyncio
async def test_panel_consensus_over_multiple_judges() -> None:
    """The comparison winner is the panel plurality of the reconciled votes."""
    async def judge(first: str, second: str, model: str) -> Prediction:
        # judge-3 is a first-slot picker (will flip out); the other two are consistent.
        if model == 'judge-3':
            return Prediction(value='A', explanation='first')
        return Prediction(value='A' if first == 'GOOD' else 'B', explanation='quality')

    comparison = await run_pairwise(
        judge_fn=judge,
        panel=['judge-1', 'judge-2', 'judge-3'],
        response_a='GOOD',
        response_b='BAD',
    )

    votes = {v.model: v for v in comparison.votes}
    assert votes['judge-1'].vote == 'A'
    assert votes['judge-2'].vote == 'A'
    assert votes['judge-3'].vote is None  # flipped
    assert comparison.winner == 'A'
