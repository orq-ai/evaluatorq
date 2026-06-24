from __future__ import annotations

import pytest

from evaluatorq.common.jury import (
    Prediction,
    VerdictKind,
    run_jury,
    validate_aggregator,
)
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
        aggregator='median',
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
        aggregator='median',
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


# --- aggregator strategies -------------------------------------------------


def _numeric_panel(scores: dict[str, float]):
    async def judge(model: str) -> Prediction:
        return Prediction(value=scores[model], explanation=f'{model}={scores[model]}')

    return judge, list(scores)


@pytest.mark.asyncio
async def test_numeric_default_is_mean() -> None:
    judge, panel = _numeric_panel({'a': 0.0, 'b': 0.6, 'c': 0.9})
    result = await run_jury(judge_fn=judge, panel=panel, verdict_kind=VerdictKind.NUMERIC)
    assert result.verdict == pytest.approx(0.5)
    # std rides along regardless of strategy
    assert result.jury.stats is not None and result.jury.stats.std > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('aggregator', 'expected'),
    [('mean_std', 0.5), ('median', 0.6), ('min', 0.0), ('max', 0.9)],
)
async def test_numeric_aggregators(aggregator: str, expected: float) -> None:
    judge, panel = _numeric_panel({'a': 0.0, 'b': 0.6, 'c': 0.9})
    result = await run_jury(
        judge_fn=judge, panel=panel, verdict_kind=VerdictKind.NUMERIC, aggregator=aggregator
    )
    assert result.verdict == pytest.approx(expected)
    # std rides along even when the verdict isn't the mean (median/min/max)
    assert result.jury.stats is not None and result.jury.stats.std > 0


@pytest.mark.asyncio
async def test_majority_returns_verdict_when_above_half() -> None:
    values = {'a': True, 'b': True, 'c': False}

    async def judge(model: str) -> Prediction:
        return Prediction(value=values[model], explanation='')

    result = await run_jury(
        judge_fn=judge, panel=list(values), verdict_kind=VerdictKind.CATEGORICAL, aggregator='majority'
    )
    assert result.verdict is True
    assert result.jury.inconclusive is False


@pytest.mark.asyncio
async def test_majority_inconclusive_without_strict_majority() -> None:
    # 3-way split: top value has 1/3, not >50% -> inconclusive.
    values = {'a': 'x', 'b': 'y', 'c': 'z'}

    async def judge(model: str) -> Prediction:
        return Prediction(value=values[model], explanation='')

    result = await run_jury(
        judge_fn=judge, panel=list(values), verdict_kind=VerdictKind.CATEGORICAL, aggregator='majority'
    )
    assert result.jury.inconclusive is True


@pytest.mark.asyncio
async def test_custom_callable_aggregator_sees_votes() -> None:
    values = {'a': 'red', 'b': 'blue', 'c': 'red'}

    async def judge(model: str) -> Prediction:
        return Prediction(value=values[model], explanation='')

    # Custom rule: pick the verdict of judge 'b' specifically (weighting demo).
    def pick_b(votes):
        return next((v.value for v in votes if v.model == 'b'), None)

    result = await run_jury(
        judge_fn=judge, panel=list(values), verdict_kind=VerdictKind.CATEGORICAL, aggregator=pick_b
    )
    assert result.verdict == 'blue'


def test_validate_aggregator_rejects_kind_mismatch() -> None:
    with pytest.raises(ValueError, match='categorical-only'):
        validate_aggregator('mode', VerdictKind.NUMERIC)
    with pytest.raises(ValueError, match='numeric-only'):
        validate_aggregator('median', VerdictKind.CATEGORICAL)
    with pytest.raises(ValueError, match='Unknown aggregator'):
        validate_aggregator('banana', VerdictKind.CATEGORICAL)
    # None and callables always pass
    validate_aggregator(None, VerdictKind.NUMERIC)
    validate_aggregator(lambda votes: None, VerdictKind.CATEGORICAL)


@pytest.mark.asyncio
async def test_mode_explicit_picks_most_common() -> None:
    values = {'a': 'x', 'b': 'x', 'c': 'y'}

    async def judge(model: str) -> Prediction:
        return Prediction(value=values[model], explanation='')

    result = await run_jury(
        judge_fn=judge, panel=list(values), verdict_kind=VerdictKind.CATEGORICAL, aggregator='mode'
    )
    assert result.verdict == 'x'
    assert result.jury.tie is False


@pytest.mark.asyncio
async def test_repetition_collapse_uses_numeric_keyword() -> None:
    # The numeric keyword also reduces a single judge's repetitions: max -> 0.9.
    scores = iter([0.1, 0.5, 0.9])

    async def judge(model: str) -> Prediction:
        return Prediction(value=next(scores), explanation='')

    result = await run_jury(
        judge_fn=judge,
        panel=['a'],
        repetitions=3,
        verdict_kind=VerdictKind.NUMERIC,
        aggregator='max',
    )
    assert result.verdict == pytest.approx(0.9)


def test_aggregator_registry_parity() -> None:
    # The fn registry, the kind partition, and the AggregatorName literal must
    # name exactly the same keyword set, or validation/dispatch silently drift.
    from typing import get_args

    from evaluatorq.common.jury import _AGG_KIND, _AGGREGATORS, AggregatorName

    names = set(get_args(AggregatorName))
    assert set(_AGGREGATORS) == names
    assert set(_AGG_KIND) == names


@pytest.mark.asyncio
async def test_majority_inconclusive_at_exactly_half() -> None:
    # 2 of 4 is exactly 50% — not a strict majority, so inconclusive.
    values = {'a': True, 'b': True, 'c': False, 'd': False}

    async def judge(model: str) -> Prediction:
        return Prediction(value=values[model], explanation='')

    result = await run_jury(
        judge_fn=judge, panel=list(values), verdict_kind=VerdictKind.CATEGORICAL, aggregator='majority'
    )
    assert result.jury.inconclusive is True


@pytest.mark.asyncio
async def test_custom_callable_returning_none_is_inconclusive() -> None:
    async def judge(model: str) -> Prediction:
        return Prediction(value='x', explanation='')

    result = await run_jury(
        judge_fn=judge,
        panel=['a', 'b'],
        verdict_kind=VerdictKind.CATEGORICAL,
        aggregator=lambda votes: None,  # deliberate "no consensus"
    )
    assert result.verdict is None
    assert result.jury.inconclusive is True


@pytest.mark.asyncio
async def test_custom_callable_sees_abstained_and_failed_votes() -> None:
    # The contract: a custom aggregator receives ALL votes, not just decisive.
    async def judge(model: str) -> Prediction:
        if model == 'ok':
            return Prediction(value='yes', explanation='')
        if model == 'abst':
            return Prediction(abstained=True, explanation='unsure')
        raise RuntimeError('down')

    seen: dict[str, int] = {}

    def count_votes(votes):
        seen['n'] = len(votes)
        seen['abstained'] = sum(1 for v in votes if v.abstained)
        seen['failed'] = sum(1 for v in votes if not v.success)
        return 'yes'

    result = await run_jury(
        judge_fn=judge,
        panel=['ok', 'abst', 'bad'],
        verdict_kind=VerdictKind.CATEGORICAL,
        aggregator=count_votes,
    )
    assert result.verdict == 'yes'
    assert seen['n'] == 3  # all three votes, not just the one decisive
    assert seen['abstained'] == 1
    assert seen['failed'] == 1


@pytest.mark.asyncio
async def test_repetition_collapse_min() -> None:
    scores = iter([0.9, 0.4, 0.2])

    async def judge(model: str) -> Prediction:
        return Prediction(value=next(scores), explanation='')

    result = await run_jury(
        judge_fn=judge, panel=['a'], repetitions=3, verdict_kind=VerdictKind.NUMERIC, aggregator='min'
    )
    assert result.verdict == pytest.approx(0.2)
