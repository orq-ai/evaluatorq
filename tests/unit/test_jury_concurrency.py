"""RES-1183: max_concurrency bounds the jury/pairwise judge fan-out.

Each test drives the fan-out with a judge_fn that tracks how many calls are
in flight at once; the peak is compared against the configured cap. The
sleep(0) yields let every scheduled call start before any finishes, so an
unbounded fan-out provably exceeds the cap.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeOutcome
from evaluatorq.common.jury import Prediction, as_semaphore, run_jury
from evaluatorq.llm_jury import llm_jury_pairwise
from evaluatorq.pairwise import run_pairwise

llm_jury_module = sys.modules['evaluatorq.llm_jury']


class InFlightTracker:
    """Counts concurrent judge calls and records the peak."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.total = 0

    async def track(self) -> None:
        self.active += 1
        self.total += 1
        self.peak = max(self.peak, self.active)
        # Yield twice so every sibling task gets scheduled while this call is
        # still "in flight"; without a cap the peak reaches the full fan-out.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.active -= 1


def _tracked_judge_fn(tracker: InFlightTracker, value: str = 'yes'):
    async def judge_fn(model: str) -> Prediction:
        await tracker.track()
        return Prediction(value=value, explanation='x')

    return judge_fn


def _tracked_pairwise_fn(tracker: InFlightTracker):
    async def judge_fn(first: str, second: str, model: str) -> Prediction:
        await tracker.track()
        return Prediction(value='A' if first == 'GOOD' else 'B', explanation='x')

    return judge_fn


# --- as_semaphore ----------------------------------------------------------


def test_as_semaphore_none_passthrough() -> None:
    assert as_semaphore(None) is None


def test_as_semaphore_existing_semaphore_passthrough() -> None:
    sem = asyncio.Semaphore(3)
    assert as_semaphore(sem) is sem


def test_as_semaphore_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match='max_concurrency'):
        as_semaphore(0)


# --- run_jury ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_jury_unbounded_fanout_exceeds_cap() -> None:
    """Sanity check: without a cap the repetitions really do run concurrently."""
    tracker = InFlightTracker()
    await run_jury(judge_fn=_tracked_judge_fn(tracker), panel=['j1', 'j2'], repetitions=3)
    assert tracker.total == 6
    assert tracker.peak > 2


@pytest.mark.asyncio
async def test_run_jury_caps_judges_times_repetitions() -> None:
    tracker = InFlightTracker()
    await run_jury(judge_fn=_tracked_judge_fn(tracker), panel=['j1', 'j2'], repetitions=3, max_concurrency=2)
    assert tracker.total == 6
    assert tracker.peak <= 2


@pytest.mark.asyncio
async def test_run_jury_cap_covers_replacement_judges() -> None:
    tracker = InFlightTracker()

    async def judge_fn(model: str) -> Prediction:
        await tracker.track()
        if model == 'primary':
            return Prediction(error='boom')
        return Prediction(value='yes', explanation='x')

    deliberation = await run_jury(
        judge_fn=judge_fn,
        panel=['primary'],
        repetitions=4,
        replacement_judges=['stand-in'],
        max_concurrency=2,
    )
    assert tracker.peak <= 2
    assert deliberation.verdict == 'yes'


# --- run_pairwise -----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pairwise_caps_judges_times_orderings_times_repetitions() -> None:
    """Panel=3, swap, reps=2 fans out 12 calls; the cap bounds them to 4."""
    tracker = InFlightTracker()
    await run_pairwise(
        judge_fn=_tracked_pairwise_fn(tracker),
        panel=['j1', 'j2', 'j3'],
        response_a='GOOD',
        response_b='BAD',
        swap=True,
        repetitions=2,
        max_concurrency=4,
    )
    assert tracker.total == 12
    assert tracker.peak <= 4


@pytest.mark.asyncio
async def test_run_pairwise_unbounded_exceeds_cap() -> None:
    tracker = InFlightTracker()
    await run_pairwise(
        judge_fn=_tracked_pairwise_fn(tracker),
        panel=['j1', 'j2', 'j3'],
        response_a='GOOD',
        response_b='BAD',
        swap=True,
        repetitions=2,
    )
    assert tracker.peak > 4


@pytest.mark.asyncio
async def test_run_pairwise_default_verdict_unchanged_with_cap() -> None:
    """The cap only schedules calls; the reconciled verdict is identical."""
    result = await run_pairwise(
        judge_fn=_tracked_pairwise_fn(InFlightTracker()),
        panel=['j1', 'j2', 'j3'],
        response_a='GOOD',
        response_b='BAD',
        max_concurrency=1,
    )
    assert result.winner == 'A'


# --- PairwiseComparator / llm_jury_pairwise ---------------------------------


@pytest.mark.asyncio
async def test_comparator_budget_shared_across_concurrent_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """One comparator, several concurrent compare() calls, one total budget."""
    tracker = InFlightTracker()

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        await tracker.track()
        value = 'A' if kwargs['replacements']['response_a'] == 'GOOD' else 'B'
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=value, explanation='x'))

    monkeypatch.setattr(llm_jury_module, 'run_judge', fake_run_judge)

    comparator = llm_jury_pairwise(judges=['j1', 'j2'], client=object(), max_concurrency=3)
    results = await asyncio.gather(*[
        comparator.compare(question='q', response_a='GOOD', response_b='BAD') for _ in range(4)
    ])

    # 4 pairs x 2 judges x 2 orderings = 16 calls, never more than 3 at once.
    assert tracker.total == 16
    assert tracker.peak <= 3
    assert all(r.winner == 'A' for r in results)


def test_llm_jury_pairwise_rejects_non_positive_cap() -> None:
    with pytest.raises(ValueError, match='max_concurrency'):
        llm_jury_pairwise(judges=['j1'], client=object(), max_concurrency=0)


def test_comparator_survives_one_asyncio_run_per_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capped comparator works across separate event loops (one asyncio.run
    per pair). A semaphore binds to the loop that first blocks on it; without
    per-loop recreation every judge call in the second loop errors and the
    verdict silently degrades to inconclusive."""

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        await asyncio.sleep(0.001)  # force the semaphore to actually block
        value = 'A' if kwargs['replacements']['response_a'] == 'GOOD' else 'B'
        return JudgeOutcome(payload=EvaluatorResponsePayload(value=value, explanation='x'))

    monkeypatch.setattr(llm_jury_module, 'run_judge', fake_run_judge)

    # max_concurrency=1 with 2 judges x 2 orderings guarantees contention.
    comparator = llm_jury_pairwise(judges=['j1', 'j2'], client=object(), max_concurrency=1)
    first = asyncio.run(comparator.compare(question='q', response_a='GOOD', response_b='BAD'))
    second = asyncio.run(comparator.compare(question='q', response_a='GOOD', response_b='BAD'))

    assert first.winner == 'A'
    assert second.winner == 'A'
