"""Unit tests for evaluatorq.simulation.cli."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from evaluatorq.simulation.cli import (
    _auto_save_run,
    _build_simulation_run,
    _format_scorer_averages,
    _infer_target_kind,
    _sanitise_run_name,
    _write_report,
    app,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    goal_achieved: bool = True,
    turn_count: int = 3,
    rules_broken: list[str] | None = None,
    scorer_scores: dict[str, float] | None = None,
) -> Any:
    from evaluatorq.simulation.types import (
        SimulationResult,
        TerminatedBy,
        TokenUsage,
    )

    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason="done",
        goal_achieved=goal_achieved,
        goal_completion_score=1.0 if goal_achieved else 0.0,
        rules_broken=rules_broken or [],
        turn_count=turn_count,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        turn_metrics=[],
        metadata={"evaluator_scores": scorer_scores or {}},
    )


def _make_datapoints(count: int = 2) -> list[Any]:
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Datapoint,
        Persona,
        Scenario,
    )

    return [
        Datapoint(
            id=f"dp-{i}",
            persona=Persona(
                name=f"User{i}",
                patience=0.5,
                assertiveness=0.5,
                politeness=0.5,
                technical_level=0.5,
                communication_style=CommunicationStyle.formal,
                background="test background",
            ),
            scenario=Scenario(name=f"Scenario{i}", goal="achieve something"),
            user_system_prompt="You are a user.",
            first_message="Hello",
        )
        for i in range(count)
    ]


def _make_datapoints_file(tmp_path: Path, count: int = 2) -> Path:
    out = tmp_path / "datapoints.jsonl"
    lines = [dp.model_dump_json() for dp in _make_datapoints(count)]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _make_results_file(tmp_path: Path) -> Path:
    out = tmp_path / "results.jsonl"
    results = [_make_result(), _make_result(goal_achieved=False, turn_count=5)]
    out.write_text(
        "\n".join(r.model_dump_json() for r in results),
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# _sanitise_run_name
# ---------------------------------------------------------------------------


def test_sanitise_run_name_basic() -> None:
    assert _sanitise_run_name("My Run") == "my_run"


def test_sanitise_run_name_collapses_underscores() -> None:
    assert _sanitise_run_name("a  b  c") == "a_b_c"


def test_sanitise_run_name_truncates() -> None:
    long_name = "a" * 100
    assert len(_sanitise_run_name(long_name)) == 64


def test_sanitise_run_name_empty_fallback() -> None:
    assert _sanitise_run_name("") == "sim"


def test_sanitise_run_name_strips_leading_trailing_underscores() -> None:
    assert not _sanitise_run_name("__hello__").startswith("_")
    assert not _sanitise_run_name("__hello__").endswith("_")


# ---------------------------------------------------------------------------
# _infer_target_kind
# ---------------------------------------------------------------------------


def test_infer_target_kind_agent_key() -> None:
    assert _infer_target_kind(agent_key="k", vercel_url=None, openai_model=None) == "orq_deployment"


def test_infer_target_kind_vercel() -> None:
    assert _infer_target_kind(agent_key=None, vercel_url="http://x", openai_model=None) == "vercel"


def test_infer_target_kind_openai() -> None:
    assert _infer_target_kind(agent_key=None, vercel_url=None, openai_model="gpt-4o") == "openai_model"


# ---------------------------------------------------------------------------
# _format_scorer_averages
# ---------------------------------------------------------------------------


def test_format_scorer_averages_empty() -> None:
    assert _format_scorer_averages({}) == "—"


def test_format_scorer_averages_values() -> None:
    out = _format_scorer_averages({"goal_achieved": 0.75})
    assert "goal_achieved=0.75" in out


# ---------------------------------------------------------------------------
# validate-dataset command
# ---------------------------------------------------------------------------


def test_validate_dataset_valid(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path, count=3)
    result = runner.invoke(app, ["validate-dataset", str(dp_file)])
    assert result.exit_code == 0
    assert "3 valid" in result.stdout


def test_validate_dataset_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-dataset", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0


def test_validate_dataset_bad_lines(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"not": "a datapoint"}\n', encoding="utf-8")
    result = runner.invoke(app, ["validate-dataset", str(bad)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


def test_export_produces_json(tmp_path: Path) -> None:
    results_file = _make_results_file(tmp_path)
    out_file = tmp_path / "payload.json"

    with patch("evaluatorq.simulation.convert.to_open_responses") as mock_conv:
        mock_conv.side_effect = lambda r: {"role": "user", "content": "x"}
        result = runner.invoke(
            app,
            ["export", "--input", str(results_file), "--output", str(out_file)],
        )

    assert result.exit_code == 0, result.stdout
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 2


def test_export_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["export", "--input", str(tmp_path / "none.jsonl"), "--output", str(tmp_path / "out.json")],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# runs command
# ---------------------------------------------------------------------------


def test_runs_no_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", str(tmp_path / "empty")])
    assert result.exit_code == 0
    assert "No sim-runs" in result.stdout


def test_runs_lists_files(tmp_path: Path) -> None:
    runs_dir = tmp_path / "sim-runs"
    runs_dir.mkdir()
    run_data = {
        "run_name": "my-run",
        "created_at": "2026-01-01T00:00:00+00:00",
        "mode": "run",
        "target_kind": "openai_model",
        "evaluator_names": ["goal_achieved"],
        "total_results": 2,
        "scorer_averages": {"goal_achieved": 0.5},
        "results": [],
    }
    (runs_dir / "my-run_20260101-000000.json").write_text(
        json.dumps(run_data), encoding="utf-8"
    )

    result = runner.invoke(app, ["runs", str(runs_dir)])
    assert result.exit_code == 0
    assert "my-run" in result.stdout


def test_runs_skips_malformed(tmp_path: Path) -> None:
    runs_dir = tmp_path / "sim-runs"
    runs_dir.mkdir()
    (runs_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    result = runner.invoke(app, ["runs", str(runs_dir)])
    assert result.exit_code == 0
    assert "malformed" in result.output


# ---------------------------------------------------------------------------
# simulate command — target validation  (datapoints in, no generation)
# ---------------------------------------------------------------------------


def test_simulate_requires_target(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(app, ["simulate", "--datapoints", str(dp_file)])
    assert result.exit_code != 0


def test_simulate_rejects_multiple_targets(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "simulate",
            "--datapoints", str(dp_file),
            "--agent-key", "k",
            "--openai-model", "gpt-4o",
        ],
        env={"ORQ_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


def test_simulate_missing_datapoints_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "simulate",
            "--datapoints", str(tmp_path / "no.jsonl"),
            "--openai-model", "gpt-4o",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


def test_simulate_unknown_evaluator(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "simulate",
            "--datapoints", str(dp_file),
            "--openai-model", "gpt-4o",
            "--evaluator", "nonexistent_evaluator_xyz",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# simulate command — success path (mocked _simulate_impl)
# ---------------------------------------------------------------------------


def test_simulate_success_no_save(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = results

        result = runner.invoke(
            app,
            [
                "simulate",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "1 simulations" in result.stdout


def test_simulate_writes_output_file(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    out_file = tmp_path / "out.jsonl"
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
        patch("evaluatorq.simulation.utils.dataset_export.export_results_to_jsonl") as mock_export,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = results
        mock_export.return_value = None

        result = runner.invoke(
            app,
            [
                "simulate",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--output", str(out_file),
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mock_export.assert_called_once()


def test_simulate_report_output_writes_full_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dp_file = _make_datapoints_file(tmp_path)
    report = tmp_path / "out" / "report.json"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = [_make_result(scorer_scores={"goal_achieved": 1.0})]

        result = runner.invoke(
            app,
            [
                "simulate",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--report-output", str(report),
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "Report saved" in result.output
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["mode"] == "simulate"
    assert data["total_results"] == 1
    assert data["scorer_averages"]["goal_achieved"] == 1.0
    # --no-save: the auto-save run-store dir must NOT be created.
    assert not (tmp_path / ".evaluatorq" / "sim-runs").exists()


def test_run_report_output_and_autosave_both_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without --no-save, --report-output writes the explicit file AND the
    # auto-save still lands under .evaluatorq/sim-runs/ (independent sinks).
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "report.json"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = [_make_result()]

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "bot",
                "--openai-model", "gpt-4o",
                "--report-output", str(report),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert report.exists()
    assert json.loads(report.read_text())["mode"] == "run"
    run_store = list((tmp_path / ".evaluatorq" / "sim-runs").glob("*.json"))
    assert len(run_store) == 1


def test_simulate_rejects_three_targets(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "simulate",
            "--datapoints", str(dp_file),
            "--agent-key", "k",
            "--vercel-url", "http://x",
            "--openai-model", "gpt-4o",
        ],
        env={"ORQ_API_KEY": "test-key"},
    )
    # Exit 2 == typer.BadParameter (the multi-target guard), per the spec's
    # exit-code table — distinct from ValueError (1) or an uncaught crash.
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# simulate command — flag forwarding (kwarg capture via patched _simulate_impl)
# ---------------------------------------------------------------------------


def test_simulate_forwards_flags(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = []

        result = runner.invoke(
            app,
            [
                "simulate",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--sim-model", "custom-model",
                "--max-turns", "7",
                "--parallelism", "3",
                "--evaluator", "goal_achieved",
                "--name", "My Run",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_impl.call_args.kwargs
    assert kwargs["sim_model"] == "custom-model"
    assert kwargs["max_turns"] == 7
    assert kwargs["parallelism"] == 3
    assert kwargs["evaluator_names"] == ["goal_achieved"]
    assert kwargs["evaluation_name"] == "My Run"


def test_simulate_impl_forwards_sim_model_to_simulate(tmp_path: Path, monkeypatch) -> None:
    # Covers the _simulate_impl -> simulate leg (test_simulate_forwards_flags stops at _simulate_impl).
    dp_file = _make_datapoints_file(tmp_path)
    captured = {}

    async def fake_simulate(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("evaluatorq.simulation.api.simulate", fake_simulate)
    with patch("evaluatorq.simulation.cli._resolve_target") as mock_target:
        mock_target.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "simulate",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--sim-model", "custom-model",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert captured["sim_model"] == "custom-model"


def test_simulate_evaluator_absent_forwards_none(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = []

        result = runner.invoke(
            app,
            ["simulate", "--datapoints", str(dp_file), "--openai-model", "gpt-4o", "--no-save"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert mock_impl.call_args.kwargs["evaluator_names"] is None


def test_simulate_evaluator_repeated_forwards_list(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = []

        result = runner.invoke(
            app,
            [
                "simulate",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--evaluator", "goal_achieved",
                "--evaluator", "criteria_met",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert mock_impl.call_args.kwargs["evaluator_names"] == ["goal_achieved", "criteria_met"]


# ---------------------------------------------------------------------------
# _auto_save_run — run-store record + scorer aggregation
# ---------------------------------------------------------------------------


def test_auto_save_scorer_averages_mixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    results = [
        _make_result(scorer_scores={"goal_achieved": 1.0, "criteria_met": 0.5}),
        _make_result(scorer_scores={"goal_achieved": 0.0}),  # criteria_met absent here
    ]
    run = _build_simulation_run(
        run_name="agg",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved", "criteria_met"],
        results=results,
    )
    path = _auto_save_run(run=run, run_name="agg")
    data = json.loads(path.read_text())
    # goal_achieved present in both -> mean(1.0, 0.0) = 0.5
    assert data["scorer_averages"]["goal_achieved"] == 0.5
    # criteria_met present in one -> mean(0.5) = 0.5, not zero-filled
    assert data["scorer_averages"]["criteria_met"] == 0.5
    assert data["total_results"] == 2


def test_auto_save_empty_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _build_simulation_run(
        run_name="empty",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[],
    )
    path = _auto_save_run(run=run, run_name="empty")
    data = json.loads(path.read_text())
    assert data["scorer_averages"] == {}
    assert data["total_results"] == 0


def test_auto_save_collision_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _build_simulation_run(
        run_name="collide",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )
    path1 = _auto_save_run(run=run, run_name="collide")
    path2 = _auto_save_run(run=run, run_name="collide")
    assert path1 != path2
    assert path2.name.endswith("_001.json")


def test_auto_save_sanitises_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _build_simulation_run(
        run_name="weird / name",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )
    path = _auto_save_run(run=run, run_name="weird / name")
    # Raw name preserved in payload, filename sanitised.
    assert json.loads(path.read_text())["run_name"] == "weird / name"
    assert "/" not in path.name
    assert path.name.startswith("weird_name_")


def test_write_report_writes_full_run_to_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run = _build_simulation_run(
        run_name="rep",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved"],
        results=[_make_result(scorer_scores={"goal_achieved": 1.0})],
    )
    out = tmp_path / "nested" / "report.json"
    _write_report(run, out)
    assert out.exists()  # parent dir created
    data = json.loads(out.read_text())
    assert data["run_name"] == "rep"
    assert data["total_results"] == 1
    assert data["scorer_averages"]["goal_achieved"] == 1.0


# ---------------------------------------------------------------------------
# run command (generate + simulate) — target validation
# ---------------------------------------------------------------------------


def test_run_requires_target() -> None:
    result = runner.invoke(
        app,
        ["run", "--agent-description", "A helpful bot"],
    )
    assert result.exit_code != 0


def test_run_success_no_save(tmp_path: Path) -> None:
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = results

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "1 simulations" in result.stdout


def test_run_forwards_flags(tmp_path: Path) -> None:
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = [_make_result()]

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o",
                "--num-personas", "2",
                "--num-scenarios", "4",
                "--max-turns", "6",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_impl.call_args.kwargs
    assert kwargs["agent_description"] == "A helpful bot"
    assert kwargs["num_personas"] == 2
    assert kwargs["num_scenarios"] == 4
    assert kwargs["max_turns"] == 6


def test_run_runtime_error_is_clean(tmp_path: Path) -> None:
    # RuntimeError (e.g. SimulationDroppedError / no datapoints) surfaces as a
    # one-line error with exit 1, not a traceback — symmetry with generate.
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.side_effect = RuntimeError("simulation job(s) produced no result")

        result = runner.invoke(
            app,
            ["run", "--agent-description", "bot", "--openai-model", "gpt-4o", "--no-save"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_simulate_runtime_error_is_clean(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.side_effect = RuntimeError("simulation job(s) produced no result")

        result = runner.invoke(
            app,
            ["simulate", "--datapoints", str(dp_file), "--openai-model", "gpt-4o", "--no-save"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# generate command (datapoints only, no simulation)
# ---------------------------------------------------------------------------


def test_generate_requires_output(tmp_path: Path) -> None:
    # --output is required: gen-only must write the datapoints somewhere.
    result = runner.invoke(
        app,
        ["generate", "--agent-description", "A helpful bot"],
    )
    assert result.exit_code != 0


def test_generate_writes_datapoints(tmp_path: Path) -> None:
    # Exercises REAL serialization (no export mock) so the generate -> file
    # handoff is verified, not faked.
    out_file = tmp_path / "dp.jsonl"
    datapoints = _make_datapoints(3)

    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = datapoints

        result = runner.invoke(
            app,
            [
                "generate",
                "--agent-description", "A helpful bot",
                "--output", str(out_file),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "Generated 3 datapoint" in result.output
    assert out_file.exists()
    assert len([ln for ln in out_file.read_text().splitlines() if ln.strip()]) == 3


def test_generate_output_roundtrips_through_simulate_loader(tmp_path: Path) -> None:
    # The whole point of gen-only: the file `generate` writes must load back
    # via the same loader `simulate --datapoints` uses, with id + fields intact.
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl

    out_file = tmp_path / "dp.jsonl"
    datapoints = _make_datapoints(2)

    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = datapoints
        result = runner.invoke(
            app,
            ["generate", "--agent-description", "bot", "--output", str(out_file)],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    loaded = load_datapoints_from_jsonl(str(out_file))
    assert [dp.id for dp in loaded] == ["dp-0", "dp-1"]  # id round-trips (not re-fabricated)
    assert [dp.persona.name for dp in loaded] == ["User0", "User1"]
    assert [dp.scenario.name for dp in loaded] == ["Scenario0", "Scenario1"]
    assert all(dp.first_message == "Hello" for dp in loaded)


def test_generate_output_passes_validate_dataset(tmp_path: Path) -> None:
    # generate's output must validate under the tool's own validate-dataset.
    out_file = tmp_path / "dp.jsonl"
    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = _make_datapoints(2)
        gen = runner.invoke(
            app,
            ["generate", "--agent-description", "bot", "--output", str(out_file)],
            env={"OPENAI_API_KEY": "test-key"},
        )
    assert gen.exit_code == 0, gen.output

    validated = runner.invoke(app, ["validate-dataset", str(out_file)])
    assert validated.exit_code == 0, validated.output
    assert "2 valid datapoint" in validated.output


def test_generate_no_datapoints_runtime_error_is_clean(tmp_path: Path) -> None:
    # RuntimeError from generation (e.g. every persona×scenario pair failed)
    # surfaces as a one-line error, not a traceback.
    out_file = tmp_path / "dp.jsonl"
    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.side_effect = RuntimeError("first-message generation produced no datapoints")
        result = runner.invoke(
            app,
            ["generate", "--agent-description", "bot", "--output", str(out_file)],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_generate_rejects_zero_personas(tmp_path: Path) -> None:
    out_file = tmp_path / "dp.jsonl"
    result = runner.invoke(
        app,
        [
            "generate",
            "--agent-description", "bot",
            "--output", str(out_file),
            "--num-personas", "0",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


def test_generate_forwards_flags(tmp_path: Path) -> None:
    out_file = tmp_path / "dp.jsonl"
    with (
        patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl,
        patch("evaluatorq.simulation.cli._write_datapoints") as mock_write,
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
    ):
        mock_impl.return_value = [object()]
        mock_write.return_value = None

        result = runner.invoke(
            app,
            [
                "generate",
                "--agent-description", "A helpful bot",
                "--output", str(out_file),
                "--sim-model", "custom-model",
                "--num-personas", "2",
                "--num-scenarios", "4",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    # Gen-only never contacts an agent target.
    mock_target.assert_not_called()
    kwargs = mock_impl.call_args.kwargs
    assert kwargs["agent_description"] == "A helpful bot"
    assert kwargs["sim_model"] == "custom-model"
    assert kwargs["num_personas"] == 2
    assert kwargs["num_scenarios"] == 4


# ---------------------------------------------------------------------------
# --sim-model flag (renamed from --model in Task 5)
# ---------------------------------------------------------------------------


def test_run_forwards_sim_model(monkeypatch):
    from typer.testing import CliRunner

    from evaluatorq.simulation import cli as sim_cli

    captured = {}

    async def fake_generate_and_simulate(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "evaluatorq.simulation.api.generate_and_simulate", fake_generate_and_simulate
    )
    result = CliRunner().invoke(
        sim_cli.app,
        [
            "run",
            "--agent-description", "x",
            "--openai-model", "gpt-5.4-mini",
            "--sim-model", "gpt-5.4-mini",
            "--num-personas", "1",
            "--num-scenarios", "1",
            "--no-save",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code == 0, result.output
    assert captured["sim_model"] == "gpt-5.4-mini"


def test_generate_forwards_sim_model(monkeypatch):
    from typer.testing import CliRunner

    from evaluatorq.simulation import cli as sim_cli

    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("evaluatorq.simulation.api.generate", fake_generate)
    monkeypatch.setattr("evaluatorq.simulation.cli._write_datapoints", lambda *a, **k: None)
    result = CliRunner().invoke(
        sim_cli.app,
        [
            "generate",
            "--agent-description", "x",
            "--output", "dp.jsonl",
            "--sim-model", "gpt-5.4-mini",
            "--num-personas", "1",
            "--num-scenarios", "1",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code == 0, result.output
    assert captured["sim_model"] == "gpt-5.4-mini"


def test_old_model_flag_rejected(monkeypatch):
    from typer.testing import CliRunner

    from evaluatorq.simulation import cli as sim_cli

    # Typer >=0.16 renders usage errors via Rich to the real stderr, which the
    # Click test runner does not capture. Disable Rich markup so the error
    # falls back to Click's native rendering on the captured stream.
    monkeypatch.setattr(sim_cli.app, "rich_markup_mode", None)

    result = CliRunner().invoke(
        sim_cli.app,
        ["run", "--agent-description", "x", "--model", "gpt-4o"],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output or "Got unexpected" in result.output


# ---------------------------------------------------------------------------
# run --save-datapoints  (Task 3)
# ---------------------------------------------------------------------------


def test_run_save_datapoints_writes_inputs(tmp_path: Path) -> None:
    """--save-datapoints writes the simulate inputs as JSONL and echoes a status message."""
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl

    dp_file = tmp_path / "dp.jsonl"

    async def fake_generate_and_simulate(**kwargs: Any) -> list[Any]:
        emit = kwargs.get("emit_datapoints")
        if emit is not None:
            emit(_make_datapoints(2))
        return []

    with (
        patch(
            "evaluatorq.simulation.api.generate_and_simulate",
            side_effect=fake_generate_and_simulate,
        ),
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
    ):
        mock_target.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "x",
                "--openai-model", "gpt-4o",
                "--save-datapoints", str(dp_file),
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert dp_file.exists(), "Datapoints file was not created"
    loaded = load_datapoints_from_jsonl(str(dp_file))
    assert len(loaded) == 2
    assert [dp.id for dp in loaded] == ["dp-0", "dp-1"]
    # Status message echoed to stderr (mix_stderr=True is typer runner default)
    assert "Datapoints saved" in result.output


def test_run_without_save_datapoints_writes_no_file(tmp_path: Path) -> None:
    """When --save-datapoints is omitted, emit_datapoints=None is passed and no file is created."""
    captured_emit: dict[str, Any] = {}

    async def fake_generate_and_simulate(**kwargs: Any) -> list[Any]:
        captured_emit["emit"] = kwargs.get("emit_datapoints")
        return []

    with (
        patch(
            "evaluatorq.simulation.api.generate_and_simulate",
            side_effect=fake_generate_and_simulate,
        ),
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
    ):
        mock_target.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "x",
                "--openai-model", "gpt-4o",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert captured_emit["emit"] is None
    # No stray JSONL files created under tmp_path
    stray = list(tmp_path.glob("*.jsonl"))
    assert stray == [], f"Unexpected JSONL files: {stray}"
