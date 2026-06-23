from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import importlib

from evaluatorq.common.judge import JudgeOutcome, EvaluatorResponsePayload
from evaluatorq.llm_jury import llm_jury
from evaluatorq.types import DataPoint

# NOTE: patch via the real module object from sys.modules, not the dotted string
# "evaluatorq.llm_jury.X". The package re-exports the ``llm_jury`` function, which
# shadows the same-named submodule as a package attribute — so both mock's dotted
# lookup and ``import evaluatorq.llm_jury as ...`` resolve to the function on 3.10.
# importlib.import_module returns the sys.modules entry, bypassing that shadowing.
llm_jury_mod = importlib.import_module("evaluatorq.llm_jury")


def test_validation_requires_exactly_one_of_criteria_prompt():
    with pytest.raises(ValueError):
        llm_jury(name="x")  # neither
    with pytest.raises(ValueError):
        llm_jury(name="x", criteria="a", prompt="b")  # both


def test_validation_rejects_empty_judge_panel():
    # An explicit empty list is a config error, not a request for the default panel.
    with pytest.raises(ValueError):
        llm_jury(name="x", criteria="c", judges=[])
    # A blank model shorthand is likewise rejected.
    with pytest.raises(ValueError):
        llm_jury(name="x", criteria="c", model="  ")


def test_validation_labels_require_categorical():
    with pytest.raises(ValueError):
        llm_jury(name="x", criteria="c", verdict_kind="numeric", labels=["a", "b"])


def test_validation_min_successful_bounds():
    with pytest.raises(ValueError):
        llm_jury(name="x", criteria="c", judges=["m1", "m2"], min_successful_judges=3)
    with pytest.raises(ValueError):
        llm_jury(name="x", criteria="c", min_successful_judges=0)


def test_validation_threshold_within_score_range():
    with pytest.raises(ValueError):
        llm_jury(
            name="x", criteria="c", verdict_kind="numeric",
            score_range=(1.0, 5.0), threshold=0.5,  # 0.5 outside (1,5)
        )


def test_temperature_zero_warns():
    with patch.object(llm_jury_mod, "resolve_llm_client") as rc, \
         patch.object(llm_jury_mod.logger, "warning") as warn:
        rc.return_value = MagicMock(client=MagicMock())
        llm_jury(name="x", criteria="c", temperature=0.0)
    assert warn.called
    assert "temperature" in warn.call_args.args[0].lower()


def test_returns_evaluator_dict():
    with patch.object(llm_jury_mod, "resolve_llm_client") as rc:
        rc.return_value = MagicMock(client=MagicMock())
        ev = llm_jury(name="correctness", criteria="correct?", judges=["m1"])
    assert ev["name"] == "correctness"
    assert callable(ev["scorer"])


@pytest.mark.asyncio
async def test_scorer_runs_panel_and_maps_pass():
    with patch.object(llm_jury_mod, "resolve_llm_client") as rc:
        rc.return_value = MagicMock(client=MagicMock())
        ev = llm_jury(
            name="correctness", criteria="correct?",
            judges=["m1"], labels=["yes", "no"], passing_labels=["yes"],
        )

    async def fake_run_judge(**kwargs):
        return JudgeOutcome(
            payload=EvaluatorResponsePayload(value="yes", explanation="good"),
            token_usage=None, raw_content="{}",
        )

    with patch.object(llm_jury_mod, "run_judge", side_effect=fake_run_judge):
        dp = DataPoint(inputs={"q": "2+2?"}, expected_output="4")
        result = await ev["scorer"]({"data": dp, "output": "4"})

    assert not isinstance(result, dict)
    assert result.pass_ is True
    assert result.value == "yes"
