from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluatorq.common.jury import Prediction
from evaluatorq.contracts import TokenUsage
from evaluatorq.pairwise import run_pairwise
from evaluatorq.pairwise_run import PairwiseRun


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


# --- Endpoint provenance --------------------------------------------------------
#
# Only the Orq router's Responses endpoint returns a priced usage block, so the
# endpoint is what tells a comparison whose token_usage carries no cost apart from
# one that was simply never priced. Before this was threaded, `_ordering` dropped
# `deliberation.endpoint` on the floor and the saved artifact had usage but no
# provenance.


def _endpoint_judge(first_endpoint: str, swapped_endpoint: str):
    """A consistent judge served by a different endpoint in each ordering.

    The swapped ordering is the one where 'GOOD' is not in the first slot.
    """

    async def judge(first: str, second: str, model: str) -> Prediction:
        served = first_endpoint if first == 'GOOD' else swapped_endpoint
        return Prediction(
            value='A' if first == 'GOOD' else 'B',
            explanation='quality',
            endpoint=served,  # pyright: ignore[reportArgumentType]
        )

    return judge


@pytest.mark.asyncio
async def test_endpoint_is_uniform_when_both_orderings_agree() -> None:
    comparison = await run_pairwise(
        judge_fn=_endpoint_judge('responses', 'responses'),
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.endpoint == 'responses'


@pytest.mark.asyncio
async def test_endpoint_is_mixed_when_orderings_used_different_endpoints() -> None:
    """One ordering on Responses and the other fallen back to chat aggregates to
    'mixed', the same label the jury uses across judges."""
    comparison = await run_pairwise(
        judge_fn=_endpoint_judge('responses', 'chat'),
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.endpoint == 'mixed'


@pytest.mark.asyncio
async def test_endpoint_is_none_when_no_judge_recorded_one() -> None:
    """A judge_fn that never sets Prediction.endpoint leaves it None. Inventing a
    default here would claim provenance the run does not have."""
    comparison = await run_pairwise(
        judge_fn=_favorite_judge('GOOD'),
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.endpoint is None


def _panel_endpoint_judge(*, split_endpoints: dict[str, str], uniform_endpoint: str):
    """A two-judge panel: split across endpoints (by model) in the first
    ordering ('GOOD' shown first), uniform in the swapped ordering.

    Exercises the fold-of-a-fold `combine_endpoints` documents but every other
    endpoint test here (single-judge panels) never reaches: one ordering
    aggregates to 'mixed' on its own, the other to a single endpoint, and the
    two are folded together into the comparison's overall endpoint.
    """

    async def judge(first: str, second: str, model: str) -> Prediction:
        served = split_endpoints[model] if first == 'GOOD' else uniform_endpoint
        return Prediction(
            value='A' if first == 'GOOD' else 'B',
            explanation='quality',
            endpoint=served,  # pyright: ignore[reportArgumentType]
        )

    return judge


@pytest.mark.asyncio
async def test_endpoint_folds_a_mixed_ordering_with_a_uniform_one() -> None:
    """A two-judge panel split across endpoints in one ordering ('mixed') and
    uniform in the other ('responses') folds to 'mixed' overall."""
    comparison = await run_pairwise(
        judge_fn=_panel_endpoint_judge(
            split_endpoints={'judge-1': 'chat', 'judge-2': 'responses'},
            uniform_endpoint='responses',
        ),
        panel=['judge-1', 'judge-2'],
        response_a='GOOD',
        response_b='BAD',
    )

    assert comparison.endpoint == 'mixed'


@pytest.mark.asyncio
async def test_saved_pairwise_run_carries_the_endpoint(tmp_path: Path) -> None:
    """The whole point of the field: it has to survive into the artifact the
    dashboard loads, not just the in-memory comparison."""
    comparison = await run_pairwise(
        judge_fn=_endpoint_judge('responses', 'chat'),
        panel=['judge-1'],
        response_a='GOOD',
        response_b='BAD',
    )
    run = PairwiseRun(run_name='endpoint-provenance', judges=['judge-1'])
    run.add(comparison, question='Q?', response_a='GOOD', response_b='BAD')
    target = run.save(tmp_path / 'run.json')

    saved = json.loads(target.read_text(encoding='utf-8'))
    assert saved['entries'][0]['comparison']['endpoint'] == 'mixed'
