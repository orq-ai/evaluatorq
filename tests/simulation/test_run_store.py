"""Unit tests for evaluatorq.simulation.utils.run_store (public API)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.simulation.utils.run_store import (
    SIM_RUNS_DIR_NAME,
    auto_save_run,
    build_simulation_run,
    get_sim_runs_dir,
    sanitise_run_name,
    write_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    goal_achieved: bool = True,
    turn_count: int = 3,
    scorer_scores: dict[str, float] | None = None,
) -> Any:
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy

    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason="done",
        goal_achieved=goal_achieved,
        goal_completion_score=1.0 if goal_achieved else 0.0,
        rules_broken=[],
        turn_count=turn_count,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        turn_metrics=[],
        metadata={"evaluator_scores": scorer_scores or {}},
    )


# ---------------------------------------------------------------------------
# sanitise_run_name
# ---------------------------------------------------------------------------


def test_sanitise_run_name_lowercases() -> None:
    assert sanitise_run_name("MyRun") == "myrun"


def test_sanitise_run_name_illegal_chars_become_underscore() -> None:
    assert sanitise_run_name("sim:agent/run") == "sim_agent_run"


def test_sanitise_run_name_collapses_repeated_underscores() -> None:
    assert sanitise_run_name("a  b  c") == "a_b_c"


def test_sanitise_run_name_empty_falls_back_to_sim() -> None:
    assert sanitise_run_name("") == "sim"


def test_sanitise_run_name_only_illegal_chars_falls_back_to_sim() -> None:
    # All chars get replaced by "_", strip leaves "", fallback kicks in.
    assert sanitise_run_name("!!!") == "sim"


def test_sanitise_run_name_strips_leading_trailing_underscores() -> None:
    result = sanitise_run_name("__hello__")
    assert not result.startswith("_")
    assert not result.endswith("_")


def test_sanitise_run_name_truncates_at_64() -> None:
    long_name = "a" * 100
    assert len(sanitise_run_name(long_name)) == 64


# ---------------------------------------------------------------------------
# build_simulation_run
# ---------------------------------------------------------------------------


def test_build_simulation_run_aggregates_scorer_averages() -> None:
    results = [
        _make_result(scorer_scores={"goal_achieved": 1.0, "criteria_met": 0.5}),
        _make_result(scorer_scores={"goal_achieved": 0.0}),  # criteria_met absent
    ]
    run = build_simulation_run(
        run_name="test-run",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved", "criteria_met"],
        results=results,
    )
    # goal_achieved present in both -> mean(1.0, 0.0) = 0.5
    assert run.scorer_averages["goal_achieved"] == pytest.approx(0.5)
    # criteria_met present in one -> mean(0.5) = 0.5, not zero-filled
    assert run.scorer_averages["criteria_met"] == pytest.approx(0.5)
    assert run.total_results == 2


def test_build_simulation_run_skips_non_numeric_scores() -> None:
    bad_scores: dict[str, Any] = {"goal_achieved": 1.0, "bad_scorer": "not_a_number"}
    results = [
        _make_result(scorer_scores=bad_scores),
        _make_result(scorer_scores={"goal_achieved": 0.5}),
    ]
    run = build_simulation_run(
        run_name="skip-non-numeric",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved", "bad_scorer"],
        results=results,
    )
    # Non-numeric score must be silently skipped — no crash.
    assert run.scorer_averages["goal_achieved"] == pytest.approx(0.75)
    # bad_scorer only had a non-numeric entry, so it should be absent or 0.
    assert "bad_scorer" not in run.scorer_averages or run.scorer_averages["bad_scorer"] == pytest.approx(0.0)


def test_build_simulation_run_empty_results() -> None:
    run = build_simulation_run(
        run_name="empty",
        mode="simulate",
        target_kind="openai_model",
        evaluator_names=[],
        results=[],
    )
    assert run.scorer_averages == {}
    assert run.total_results == 0


def test_build_simulation_run_sets_run_name_and_mode() -> None:
    run = build_simulation_run(
        run_name="my-run",
        mode="run",
        target_kind="orq_agent",
        evaluator_names=["goal_achieved"],
        results=[_make_result(scorer_scores={"goal_achieved": 1.0})],
    )
    assert run.run_name == "my-run"
    assert run.mode == "run"
    assert run.target_kind == "orq_agent"


# ---------------------------------------------------------------------------
# auto_save_run
# ---------------------------------------------------------------------------


def test_auto_save_run_writes_json_under_sim_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: tmp_path / "sim-runs",
    )
    run = build_simulation_run(
        run_name="my-run",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )
    path = auto_save_run(run=run, run_name="my-run")

    assert path.exists()
    assert path.suffix == ".json"
    # Hyphens are kept by sanitise_run_name (only non-[a-z0-9_-] chars are replaced).
    assert path.name.startswith("my-run_")
    data = json.loads(path.read_text())
    assert data["run_name"] == "my-run"


def test_auto_save_run_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "deep" / "nested" / "sim-runs"
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: runs_dir,
    )
    run = build_simulation_run(
        run_name="nested-run",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[],
    )
    path = auto_save_run(run=run, run_name="nested-run")
    assert path.exists()


def test_auto_save_run_collision_gets_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "sim-runs"
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: runs_dir,
    )
    run = build_simulation_run(
        run_name="collide",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )
    path1 = auto_save_run(run=run, run_name="collide")
    path2 = auto_save_run(run=run, run_name="collide")

    assert path1 != path2
    assert path2.name.endswith("_001.json")


def test_auto_save_run_sanitises_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: tmp_path / "sim-runs",
    )
    run = build_simulation_run(
        run_name="weird / name",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )
    path = auto_save_run(run=run, run_name="weird / name")

    # Raw name preserved in the payload, filename is sanitised.
    assert json.loads(path.read_text())["run_name"] == "weird / name"
    assert "/" not in path.name
    assert path.name.startswith("weird_name_")


def test_auto_save_run_scorer_averages_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: tmp_path / "sim-runs",
    )
    results = [
        _make_result(scorer_scores={"goal_achieved": 1.0, "criteria_met": 0.5}),
        _make_result(scorer_scores={"goal_achieved": 0.0}),
    ]
    run = build_simulation_run(
        run_name="agg",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved", "criteria_met"],
        results=results,
    )
    path = auto_save_run(run=run, run_name="agg")
    data = json.loads(path.read_text())

    assert data["scorer_averages"]["goal_achieved"] == pytest.approx(0.5)
    assert data["scorer_averages"]["criteria_met"] == pytest.approx(0.5)
    assert data["total_results"] == 2


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


def test_write_report_writes_json_to_explicit_path(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name="rep",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved"],
        results=[_make_result(scorer_scores={"goal_achieved": 1.0})],
    )
    out = tmp_path / "report.json"
    write_report(run, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["run_name"] == "rep"
    assert data["total_results"] == 1
    assert data["scorer_averages"]["goal_achieved"] == pytest.approx(1.0)


def test_write_report_creates_parent_dirs(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name="nested-rep",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[],
    )
    out = tmp_path / "nested" / "deeply" / "report.json"
    write_report(run, out)

    assert out.exists()


def test_write_report_full_run_json(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name="full-rep",
        mode="simulate",
        target_kind="orq_agent",
        evaluator_names=["goal_achieved"],
        results=[
            _make_result(scorer_scores={"goal_achieved": 1.0}),
            _make_result(goal_achieved=False, scorer_scores={"goal_achieved": 0.0}),
        ],
    )
    out = tmp_path / "subdir" / "full.json"
    write_report(run, out)

    data = json.loads(out.read_text())
    assert data["mode"] == "simulate"
    assert data["target_kind"] == "orq_agent"
    assert data["total_results"] == 2
    assert data["scorer_averages"]["goal_achieved"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_sim_runs_dir / SIM_RUNS_DIR_NAME
# ---------------------------------------------------------------------------


def test_get_sim_runs_dir_returns_cwd_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drop the autouse EVALUATORQ_DIR override so the cwd fallback is exercised.
    monkeypatch.delenv("EVALUATORQ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = get_sim_runs_dir()
    expected = tmp_path / SIM_RUNS_DIR_NAME
    assert result == expected


def test_sim_runs_dir_name_is_path() -> None:
    assert isinstance(SIM_RUNS_DIR_NAME, Path)


# ---------------------------------------------------------------------------
# SDK save wiring: simulate(save=..., run_output=...) through _simulate_core
# (plan Verification §3). Patches the choke point _simulate_via_evaluatorq so
# no LLM/runner is needed — exercises the save gate, target_kind, and branch.
# ---------------------------------------------------------------------------


def _make_datapoint() -> Any:
    from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario, SimulationDatapoint

    persona = Persona(
        name="T",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="t",
    )
    return SimulationDatapoint(
        id="dp-1",
        persona=persona,
        scenario=Scenario(name="S", goal="help"),
        user_system_prompt="sys",
        first_message="hi",
    )


async def _run_simulate(*, runs_dir: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    from unittest.mock import AsyncMock, patch

    from evaluatorq.simulation.api import simulate

    monkeypatch.setattr("evaluatorq.simulation.utils.run_store.get_sim_runs_dir", lambda: runs_dir)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]
    with patch("evaluatorq.simulation.api._simulate_via_evaluatorq", AsyncMock(return_value=results)):
        await simulate(
            target=lambda _messages: "hi",
            datapoints=[_make_datapoint()],
            sim_model="test",
            upload_results=False,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_simulate_save_true_writes_one_run_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _run_simulate(runs_dir=tmp_path, monkeypatch=monkeypatch, save=True)
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_simulate_save_true_callable_target_kind_is_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a plain-callable target produced target_kind='callback', which
    # was not in the SimulationRun.target_kind Literal and crashed on save=True.
    await _run_simulate(runs_dir=tmp_path, monkeypatch=monkeypatch, save=True)
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["target_kind"] == "callback"
    assert data["mode"] == "simulate"


@pytest.mark.asyncio
async def test_simulate_save_false_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _run_simulate(runs_dir=tmp_path, monkeypatch=monkeypatch, save=False)
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_simulate_run_output_writes_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    explicit = tmp_path / "nested" / "explicit.json"
    await _run_simulate(runs_dir=auto_dir, monkeypatch=monkeypatch, save=True, run_output=str(explicit))
    assert explicit.exists()
    assert list(auto_dir.glob("*.json")) == []  # explicit path bypasses auto-save dir


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    # OSError = disk/perms, RuntimeError = collision exhaustion, ValueError =
    # pydantic serialization (PydanticSerializationError <: ValueError).
    [OSError("disk full"), RuntimeError("no free filename"), ValueError("cannot serialize")],
)
async def test_simulate_save_failure_still_returns_results(
    error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A persistence failure must not discard a completed, paid-for run.
    from unittest.mock import AsyncMock, patch

    from evaluatorq.simulation.api import simulate

    monkeypatch.setattr("evaluatorq.simulation.utils.run_store.get_sim_runs_dir", lambda: tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]

    def _boom(**_kwargs: Any) -> Path:
        raise error

    with (
        patch("evaluatorq.simulation.api._simulate_via_evaluatorq", AsyncMock(return_value=results)),
        patch("evaluatorq.simulation.api.auto_save_run", _boom),
    ):
        out = await simulate(
            target=lambda _messages: "hi",
            datapoints=[_make_datapoint()],
            sim_model="test",
            upload_results=False,
            save=True,
        )

    assert out == results  # results survived the save failure
    assert list(tmp_path.glob("*.json")) == []  # nothing written
