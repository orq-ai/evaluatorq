"""CyclicJudge (round-robin) judge assignment for llm_jury and llm_jury_pairwise.

Pins the paper's balanced-design requirement (arXiv:2603.01865): under
assignment='cyclic' every item gets exactly one judge, judges rotate in order,
and each judge covers an equal share of the run even under concurrency.
"""

from __future__ import annotations

import asyncio
import importlib
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeOutcome
from evaluatorq.llm_jury import llm_jury, llm_jury_pairwise
from evaluatorq.pairwise import build_report
from evaluatorq.types import DataPoint

llm_jury_mod = importlib.import_module("evaluatorq.llm_jury")


def _fake_run_judge(calls: list[str], value: str = "yes"):
    async def fake(**kwargs):
        calls.append(kwargs["model"])
        return JudgeOutcome(
            payload=EvaluatorResponsePayload(value=value, explanation="ok"),
            token_usage=None,
            raw_content="{}",
        )

    return fake


def _datapoint() -> DataPoint:
    return DataPoint(inputs={"q": "?"}, expected_output="x")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_assignment_raises():
    with pytest.raises(ValueError, match="unknown assignment"):
        llm_jury(name="x", criteria="c", judges=["a", "b"], assignment="random")  # pyright: ignore[reportArgumentType]


def test_cyclic_requires_min_successful_judges_1():
    with pytest.raises(ValueError, match="min_successful_judges"):
        llm_jury(name="x", criteria="c", judges=["a", "b"], assignment="cyclic", min_successful_judges=2)
    with pytest.raises(ValueError, match="min_successful_judges"):
        llm_jury_pairwise(judges=["a", "b"], assignment="cyclic", min_successful_judges=2)


# ---------------------------------------------------------------------------
# Scoring jury
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cyclic_rotates_one_judge_per_datapoint():
    ev = llm_jury(
        name="x",
        criteria="c",
        judges=["m1", "m2", "m3"],
        assignment="cyclic",
        labels=["yes", "no"],
        passing_labels=["yes"],
        client=MagicMock(),
    )
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        for _ in range(6):
            result = await ev["scorer"]({"data": _datapoint(), "output": "x"})
            assert not isinstance(result, dict)
            assert result.pass_ is True
            assert result.value == "yes"
    # One judge per item, strict round-robin over the panel.
    assert calls == ["m1", "m2", "m3", "m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_all_assignment_still_runs_full_panel():
    ev = llm_jury(name="x", criteria="c", judges=["m1", "m2", "m3"], client=MagicMock())
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        await ev["scorer"]({"data": _datapoint(), "output": "x"})
    assert sorted(calls) == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_cyclic_balances_exactly_under_concurrency():
    ev = llm_jury(
        name="x", criteria="c", judges=["m1", "m2", "m3"], assignment="cyclic", client=MagicMock()
    )
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        await asyncio.gather(*(ev["scorer"]({"data": _datapoint(), "output": "x"}) for _ in range(9)))
    # Interleaving may reorder, but shares stay exactly balanced.
    assert Counter(calls) == {"m1": 3, "m2": 3, "m3": 3}


@pytest.mark.asyncio
async def test_cyclic_single_model_panel_degenerates_to_single_judge():
    ev = llm_jury(name="x", criteria="c", judges=["only"], assignment="cyclic", client=MagicMock())
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        for _ in range(3):
            await ev["scorer"]({"data": _datapoint(), "output": "x"})
    assert calls == ["only", "only", "only"]


@pytest.mark.asyncio
async def test_cyclic_rotates_deduplicated_panel():
    # Duplicate judges must not skew the rotation shares.
    ev = llm_jury(
        name="x", criteria="c", judges=["m1", "m1", "m2"], assignment="cyclic", client=MagicMock()
    )
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        for _ in range(4):
            await ev["scorer"]({"data": _datapoint(), "output": "x"})
    assert calls == ["m1", "m2", "m1", "m2"]


# ---------------------------------------------------------------------------
# Pairwise jury
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairwise_cyclic_one_judge_per_pair_with_swap():
    comparator = llm_jury_pairwise(
        judges=["a", "b", "c"], assignment="cyclic", client=MagicMock()
    )
    calls: list[str] = []

    async def fake_prefers_x(**kwargs):
        # A position-consistent judge: prefers response 'x' whichever slot it
        # is in, so the swapped ordering reconciles instead of cancelling.
        calls.append(kwargs["model"])
        value = "A" if kwargs["replacements"]["response_a"] == "x" else "B"
        return JudgeOutcome(
            payload=EvaluatorResponsePayload(value=value, explanation="ok"),
            token_usage=None,
            raw_content="{}",
        )

    comparisons = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=fake_prefers_x):
        for _ in range(6):
            comparisons.append(await comparator.compare(question="q", response_a="x", response_b="y"))

    # Each comparison carries exactly one judge's vote, rotating in order.
    per_pair_models = [[v.model for v in c.votes] for c in comparisons]
    assert per_pair_models == [["a"], ["b"], ["c"], ["a"], ["b"], ["c"]]
    # Swap stays per-judge: both orderings ran, so 2 calls per pair.
    assert Counter(calls) == {"a": 4, "b": 4, "c": 4}
    # The report still rolls up per-judge stats across the cyclic run.
    report = build_report(comparisons)
    assert sorted(stats.model for stats in report.per_judge) == ["a", "b", "c"]
    assert report.a_win_rate == 1.0
