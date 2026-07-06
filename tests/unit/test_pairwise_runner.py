from __future__ import annotations

import pytest

from evaluatorq.common.jury import Prediction
from evaluatorq.contracts import TokenUsage
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


@pytest.mark.asyncio
async def test_min_successful_judges_gates_the_winner() -> None:
    """A quorum shortfall is inconclusive even when the lone survivor is decisive (#1)."""

    async def judge(first: str, second: str, model: str) -> Prediction:
        if model in ('judge-2', 'judge-3'):
            raise RuntimeError('judge outage')
        return Prediction(value='A' if first == 'GOOD' else 'B', explanation='quality')

    comparison = await run_pairwise(
        judge_fn=judge,
        panel=['judge-1', 'judge-2', 'judge-3'],
        response_a='GOOD',
        response_b='BAD',
        min_successful_judges=3,
    )

    # judge-1 alone is decisive, but 1 < 3 required, so the panel can't decide.
    assert comparison.winner == 'inconclusive'


@pytest.mark.asyncio
async def test_replacement_judge_casts_a_reconciled_vote_under_swap() -> None:
    """A stand-in runs in both orderings and its reconciled vote counts (#2)."""

    async def judge(first: str, second: str, model: str) -> Prediction:
        if model == 'primary':
            raise RuntimeError('primary down in every ordering')
        return Prediction(value='A' if first == 'GOOD' else 'B', explanation='backup quality')

    comparison = await run_pairwise(
        judge_fn=judge,
        panel=['primary'],
        response_a='GOOD',
        response_b='BAD',
        replacement_judges=['backup'],
    )

    votes = {v.model: v for v in comparison.votes}
    assert votes['backup'].vote == 'A'
    assert votes['backup'].replacement is True
    assert comparison.winner == 'A'


@pytest.mark.asyncio
async def test_repetitions_run_each_ordering() -> None:
    """repetitions>1 fans out per ordering and still reconciles to one vote."""
    calls = 0

    async def judge(first: str, second: str, model: str) -> Prediction:
        nonlocal calls
        calls += 1
        return Prediction(value='A' if first == 'GOOD' else 'B', explanation='quality')

    comparison = await run_pairwise(
        judge_fn=judge,
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
        repetitions=3,
    )

    assert calls == 6  # 3 repetitions x 2 orderings
    assert comparison.votes[0].vote == 'A'
    assert comparison.votes[0].flipped is False


@pytest.mark.asyncio
async def test_non_pairwise_verdict_is_rejected() -> None:
    """A judge_fn leaking a non-A/B/tie value raises rather than surfacing 'True' as a winner."""

    async def judge(first: str, second: str, model: str) -> Prediction:
        return Prediction(value=True, explanation='oops, a bool')

    with pytest.raises(ValueError, match='expected'):
        await run_pairwise(
            judge_fn=judge,
            panel=['judge-1'],
            response_a='GOOD',
            response_b='BAD',
        )


@pytest.mark.asyncio
async def test_token_usage_is_summed_across_orderings() -> None:
    """Per-call token usage is aggregated onto the comparison (cost-accounting parity)."""

    async def judge(first: str, second: str, model: str) -> Prediction:
        return Prediction(
            value='A' if first == 'GOOD' else 'B',
            explanation='quality',
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    comparison = await run_pairwise(
        judge_fn=judge,
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.token_usage is not None
    # One judge, two orderings -> two calls summed.
    assert comparison.token_usage.input_tokens == 20
    assert comparison.token_usage.output_tokens == 10
