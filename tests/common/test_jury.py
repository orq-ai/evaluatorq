from __future__ import annotations

import pytest

from evaluatorq.common.jury import Prediction, VerdictKind, run_jury
from evaluatorq.contracts import TokenUsage


@pytest.mark.asyncio
async def test_repetitions_failed_populated_on_partial_error() -> None:
    """When some repetitions error and others succeed, repetitions_failed is > 0."""
    call_count = 0

    async def judge(model: str) -> Prediction:
        nonlocal call_count
        call_count += 1
        # First call errors, second succeeds
        if call_count == 1:
            raise RuntimeError('simulated error')
        return Prediction(value=True, explanation='ok')

    result = await run_jury(
        judge_fn=judge,
        panel=['judge-a'],
        repetitions=2,
    )

    assert len(result.jury.votes) == 1
    vote = result.jury.votes[0]
    # One rep failed, one succeeded — judge overall succeeded with value=True
    assert vote.success is True
    assert vote.value is True
    assert vote.repetitions_failed == 1


@pytest.mark.asyncio
async def test_repetitions_failed_zero_when_all_succeed() -> None:
    """When all repetitions succeed, repetitions_failed is 0."""

    async def judge(model: str) -> Prediction:
        return Prediction(value=True, explanation='ok')

    result = await run_jury(
        judge_fn=judge,
        panel=['judge-a'],
        repetitions=3,
    )

    assert result.jury.votes[0].repetitions_failed == 0


@pytest.mark.asyncio
async def test_categorical_tie_uses_caller_tie_break() -> None:
    values = {'a': True, 'b': False}

    async def judge(model: str) -> Prediction:
        return Prediction(value=values[model], explanation=f'{model} says {values[model]}')

    result = await run_jury(
        judge_fn=judge,
        panel=['a', 'b'],
        verdict_kind=VerdictKind.CATEGORICAL,
        tie_break=lambda _values: False,
    )

    assert result.verdict is False
    assert result.jury.tie is True
    assert result.jury.raw_agreement == 0.5


@pytest.mark.asyncio
async def test_abstain_is_successful_but_not_decisive() -> None:
    async def judge(model: str) -> Prediction:
        if model == 'abstain':
            return Prediction(abstained=True, explanation='not enough evidence')
        return Prediction(value=True, explanation='decisive')

    result = await run_jury(
        judge_fn=judge,
        panel=['abstain', 'decisive'],
        min_successful_judges=2,
    )

    assert result.verdict is None
    assert result.jury.inconclusive is True
    assert result.jury.judges_succeeded == 1
    assert result.jury.votes[0].success is True
    assert result.jury.votes[0].abstained is True


@pytest.mark.asyncio
async def test_numeric_mean_and_usage_sum() -> None:
    values = {'a': 0.2, 'b': 0.8}

    async def judge(model: str) -> Prediction:
        return Prediction(
            value=values[model],
            explanation=model,
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3, calls=1),
        )

    result = await run_jury(
        judge_fn=judge,
        panel=['a', 'b'],
        verdict_kind=VerdictKind.NUMERIC,
    )

    assert result.verdict == 0.5
    assert result.jury.stats is not None
    assert result.jury.stats.mean == 0.5
    assert result.jury.raw_agreement is None
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 6


@pytest.mark.asyncio
async def test_numeric_median_aggregation() -> None:
    """Median aggregation: three judges [0.1, 0.5, 0.9] → median 0.5."""
    vals = {'a': 0.1, 'b': 0.5, 'c': 0.9}

    async def judge(model: str) -> Prediction:
        return Prediction(value=vals[model], explanation=model)

    result = await run_jury(
        judge_fn=judge,
        panel=['a', 'b', 'c'],
        verdict_kind=VerdictKind.NUMERIC,
        numeric_aggregation='median',
    )

    assert result.verdict == pytest.approx(0.5)
    assert result.jury.stats is not None
    assert result.jury.stats.mean == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_numeric_median_even_count() -> None:
    """Even count median: [0.2, 0.8] → (0.2 + 0.8) / 2 = 0.5."""
    vals = {'a': 0.2, 'b': 0.8}

    async def judge(model: str) -> Prediction:
        return Prediction(value=vals[model], explanation=model)

    result = await run_jury(
        judge_fn=judge,
        panel=['a', 'b'],
        verdict_kind=VerdictKind.NUMERIC,
        numeric_aggregation='median',
    )

    assert result.verdict == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_duplicate_replacement_judges_deduped() -> None:
    """A repeated stand-in must not cast two independent replacement votes."""
    calls: list[str] = []

    async def judge(model: str) -> Prediction:
        calls.append(model)
        if model == 'primary':
            return Prediction(error='primary down')
        return Prediction(value=True, explanation=model)

    result = await run_jury(
        judge_fn=judge,
        panel=['primary'],
        replacement_judges=['stand-in', 'stand-in'],
    )

    # Only one unique replacement is summoned despite the duplicate in the list.
    assert calls.count('stand-in') == 1
    assert result.jury.replacements_used == 1


@pytest.mark.asyncio
async def test_propagate_errors_aborts_for_lone_judge() -> None:
    """With no redundancy, an infra error propagates instead of degrading."""

    async def judge(model: str) -> Prediction:
        raise RuntimeError('api down')

    with pytest.raises(RuntimeError, match='api down'):
        await run_jury(judge_fn=judge, panel=['only'], propagate_errors=True)


@pytest.mark.asyncio
async def test_errors_swallowed_when_not_propagating() -> None:
    """Default (panel has redundancy): a judge error degrades to a failed vote."""

    async def judge(model: str) -> Prediction:
        raise RuntimeError('api down')

    result = await run_jury(judge_fn=judge, panel=['only'], propagate_errors=False)
    assert result.jury.votes[0].success is False
    assert result.jury.inconclusive is True
