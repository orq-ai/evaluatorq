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
    _format_scorer_averages,
    _infer_target_kind,
    _sanitise_run_name,
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


def _make_datapoints_file(tmp_path: Path, count: int = 2) -> Path:
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Datapoint,
        Persona,
        Scenario,
    )

    out = tmp_path / "datapoints.jsonl"
    lines = []
    for i in range(count):
        dp = Datapoint(
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
        lines.append(dp.model_dump_json())
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
# run command — target validation
# ---------------------------------------------------------------------------


def test_run_requires_target(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(app, ["run", "--datapoints", str(dp_file)])
    assert result.exit_code != 0


def test_run_rejects_multiple_targets(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "--datapoints", str(dp_file),
            "--agent-key", "k",
            "--openai-model", "gpt-4o",
        ],
        env={"ORQ_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


def test_run_missing_datapoints_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--datapoints", str(tmp_path / "no.jsonl"),
            "--openai-model", "gpt-4o",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


def test_run_unknown_evaluator(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "--datapoints", str(dp_file),
            "--openai-model", "gpt-4o",
            "--evaluator", "nonexistent_evaluator_xyz",
        ],
        env={"OPENAI_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run command — success path (mocked simulate)
# ---------------------------------------------------------------------------


def test_run_success_no_save(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]

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
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "1 simulations" in result.stdout


def test_run_writes_output_file(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    out_file = tmp_path / "out.jsonl"
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
        patch("evaluatorq.simulation.utils.dataset_export.export_results_to_jsonl") as mock_export,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = results
        mock_export.return_value = None

        result = runner.invoke(
            app,
            [
                "run",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--output", str(out_file),
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output


def test_run_rejects_three_targets(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
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
# run command — flag forwarding (kwarg capture via patched _run_impl)
# ---------------------------------------------------------------------------


def test_run_forwards_flags(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = []

        result = runner.invoke(
            app,
            [
                "run",
                "--datapoints", str(dp_file),
                "--openai-model", "gpt-4o",
                "--model", "custom-model",
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
    assert kwargs["model"] == "custom-model"
    assert kwargs["max_turns"] == 7
    assert kwargs["parallelism"] == 3
    assert kwargs["evaluator_names"] == ["goal_achieved"]
    assert kwargs["evaluation_name"] == "My Run"


def test_run_evaluator_absent_forwards_none(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = []

        result = runner.invoke(
            app,
            ["run", "--datapoints", str(dp_file), "--openai-model", "gpt-4o", "--no-save"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert mock_impl.call_args.kwargs["evaluator_names"] is None


def test_run_evaluator_repeated_forwards_list(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = []

        result = runner.invoke(
            app,
            [
                "run",
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
    path = _auto_save_run(
        run_name="agg",
        mode="run",
        target_kind="openai_model",
        evaluator_names=["goal_achieved", "criteria_met"],
        results=results,
    )
    data = json.loads(path.read_text())
    # goal_achieved present in both -> mean(1.0, 0.0) = 0.5
    assert data["scorer_averages"]["goal_achieved"] == 0.5
    # criteria_met present in one -> mean(0.5) = 0.5, not zero-filled
    assert data["scorer_averages"]["criteria_met"] == 0.5
    assert data["total_results"] == 2


def test_auto_save_empty_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = _auto_save_run(
        run_name="empty",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[],
    )
    data = json.loads(path.read_text())
    assert data["scorer_averages"] == {}
    assert data["total_results"] == 0


def test_auto_save_collision_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    results = [_make_result()]
    kwargs = {
        "run_name": "collide",
        "mode": "run",
        "target_kind": "openai_model",
        "evaluator_names": [],
        "results": results,
    }
    path1 = _auto_save_run(**kwargs)  # type: ignore[arg-type]
    path2 = _auto_save_run(**kwargs)  # type: ignore[arg-type]
    assert path1 != path2
    assert path2.name.endswith("_001.json")


def test_auto_save_sanitises_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = _auto_save_run(
        run_name="weird / name",
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )
    # Raw name preserved in payload, filename sanitised.
    assert json.loads(path.read_text())["run_name"] == "weird / name"
    assert "/" not in path.name
    assert path.name.startswith("weird_name_")


# ---------------------------------------------------------------------------
# generate command — target validation
# ---------------------------------------------------------------------------


def test_generate_requires_target() -> None:
    result = runner.invoke(
        app,
        ["generate", "--agent-description", "A helpful bot"],
    )
    assert result.exit_code != 0


def test_generate_success_no_save(tmp_path: Path) -> None:
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = results

        result = runner.invoke(
            app,
            [
                "generate",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "1 simulations" in result.stdout


def test_generate_forwards_flags(tmp_path: Path) -> None:
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = [_make_result()]

        result = runner.invoke(
            app,
            [
                "generate",
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
