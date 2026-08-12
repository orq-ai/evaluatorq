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
        # Yield to the event loop so asyncio.gather genuinely interleaves:
        # without this the concurrency tests cannot distinguish arrival order
        # from dataset order and would pass whatever the assignment logic does.
        await asyncio.sleep(0)
        return JudgeOutcome(
            payload=EvaluatorResponsePayload(value=value, explanation="ok"),
            token_usage=None,
            raw_content="{}",
        )

    return fake


def _fake_run_judge_failing(calls: list[str], fail_model: str, value: str = "yes"):
    async def fake(**kwargs):
        calls.append(kwargs["model"])
        await asyncio.sleep(0)
        if kwargs["model"] == fail_model:
            raise RuntimeError("boom")
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


@pytest.mark.asyncio
async def test_cyclic_failed_item_degrades_to_inconclusive():
    # A judge that fails mechanically must not raise out of the scorer; with no
    # replacements configured, only that judge's item comes back inconclusive.
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
    outcomes = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge_failing(calls, "m2")):
        for _ in range(3):
            result = await ev["scorer"]({"data": _datapoint(), "output": "x"})
            assert not isinstance(result, dict)
            outcomes.append((result.value, result.pass_))
    assert outcomes == [("yes", True), ("inconclusive", None), ("yes", True)]
    assert calls == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_cyclic_failed_judge_promotes_replacement():
    # A configured replacement stands in for the failed judge and casts a real
    # vote, so the item stays conclusive.
    ev = llm_jury(
        name="x",
        criteria="c",
        judges=["m1", "m2", "m3"],
        assignment="cyclic",
        replacement_judges=["r1"],
        labels=["yes", "no"],
        passing_labels=["yes"],
        client=MagicMock(),
    )
    calls: list[str] = []
    outcomes = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge_failing(calls, "m2")):
        for _ in range(3):
            result = await ev["scorer"]({"data": _datapoint(), "output": "x"})
            assert not isinstance(result, dict)
            outcomes.append((result.value, result.pass_))
    assert outcomes == [("yes", True), ("yes", True), ("yes", True)]
    assert calls == ["m1", "m2", "r1", "m3"]


@pytest.mark.asyncio
async def test_cyclic_item_reports_no_agreement_stats():
    # One vote has no cross-judge agreement: the summary must say n/a instead
    # of rendering a fake 100% indistinguishable from a unanimous panel.
    ev = llm_jury(
        name="x",
        criteria="c",
        judges=["m1", "m2", "m3"],
        assignment="cyclic",
        labels=["yes", "no"],
        passing_labels=["yes"],
        client=MagicMock(),
    )
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge([])):
        result = await ev["scorer"]({"data": _datapoint(), "output": "x"})
    assert not isinstance(result, dict)
    assert result.explanation is not None
    assert "raw agreement n/a" in result.explanation

    # assignment='all' keeps reporting the real cross-judge rate.
    ev_all = llm_jury(
        name="x",
        criteria="c",
        judges=["m1", "m2", "m3"],
        labels=["yes", "no"],
        passing_labels=["yes"],
        client=MagicMock(),
    )
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge([])):
        result_all = await ev_all["scorer"]({"data": _datapoint(), "output": "x"})
    assert not isinstance(result_all, dict)
    assert result_all.explanation is not None
    assert "raw agreement 100%" in result_all.explanation


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
        value = "A" if kwargs["replacements"]["response_a"]["output"]["response"] == "x" else "B"
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


# ---------------------------------------------------------------------------
# Row-keyed assignment (the evaluatorq() runner path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cyclic_maps_judge_by_dataset_row_not_arrival_order():
    """Inside the runner the mapping is row % len(panel), not call order.

    Rows arrive out of order on purpose: 3 -> m1 (3%3=0), 1 -> m2, 4 -> m2.
    An arrival-order cursor would produce m1, m2, m3 here instead.
    """
    ev = llm_jury(
        name="x", criteria="c", judges=["m1", "m2", "m3"], assignment="cyclic", client=MagicMock()
    )
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        for row in (3, 1, 4):
            await ev["scorer"]({"data": _datapoint(), "output": "x", "row": row})
    assert calls == ["m1", "m2", "m2"]


@pytest.mark.asyncio
async def test_cyclic_row_mapping_is_parallelism_independent():
    """Concurrent, shuffled rows still land on judge row % len(panel).

    The judge identity is read back from raw_output, so this asserts the full
    item->judge mapping, not just the share balance.
    """
    ev = llm_jury(
        name="x", criteria="c", judges=["m1", "m2", "m3"], assignment="cyclic", client=MagicMock()
    )
    rows = [4, 0, 7, 2, 5, 8, 1, 6, 3]
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        results = await asyncio.gather(
            *(ev["scorer"]({"data": _datapoint(), "output": "x", "row": row}) for row in rows)
        )
    panel = ["m1", "m2", "m3"]
    for row, result in zip(rows, results):
        assert not isinstance(result, dict)
        assert result.raw_output is not None
        votes = result.raw_output["jury"]["votes"]
        assert [v["model"] for v in votes] == [panel[row % 3]]
    assert Counter(calls) == {"m1": 3, "m2": 3, "m3": 3}


@pytest.mark.asyncio
async def test_cyclic_single_model_panel_propagates_judge_failure():
    """A lone cyclic judge with no stand-ins has no redundancy: an outage must
    raise out of the scorer, never degrade to inconclusive. This is the
    propagate_errors contract (len(deduped) == 1 and no replacements); a
    regression that swaps deduped for the always-length-1 run panel would make
    every cyclic run swallow outages silently."""
    ev = llm_jury(name="x", criteria="c", judges=["only"], assignment="cyclic", client=MagicMock())
    calls: list[str] = []
    with (
        patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge_failing(calls, "only")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await ev["scorer"]({"data": _datapoint(), "output": "x"})


@pytest.mark.asyncio
async def test_result_carries_jury_record_for_audit():
    """raw_output['jury'] records which judge scored the item (per-judge votes),
    so the cyclic rotation is auditable from the results alone."""
    ev = llm_jury(
        name="x", criteria="c", judges=["m1", "m2", "m3"], assignment="cyclic", client=MagicMock()
    )
    calls: list[str] = []
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge(calls)):
        result = await ev["scorer"]({"data": _datapoint(), "output": "x", "row": 1})
    assert not isinstance(result, dict)
    assert result.raw_output is not None
    votes = result.raw_output["jury"]["votes"]
    assert [v["model"] for v in votes] == ["m2"]
    assert votes[0]["value"] == "yes"


@pytest.mark.asyncio
async def test_all_assignment_result_has_no_jury_record():
    """The jury record is cyclic-only: under 'all' the panel itself is the
    record, and raw_output stays None so external callers using it as a
    signal keep the pre-cyclic behavior."""
    ev = llm_jury(name="x", criteria="c", judges=["m1", "m2", "m3"], client=MagicMock())
    with patch.object(llm_jury_mod, "run_judge", side_effect=_fake_run_judge([])):
        result = await ev["scorer"]({"data": _datapoint(), "output": "x"})
    assert not isinstance(result, dict)
    assert result.raw_output is None
