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
    _configure_logging,
    _echo_using,
    _format_scorer_averages,
    _resolve_agent_description,
    _resolve_target,
    _sanitise_run_name,
    _shell_path,
    _write_report,
    app,
)
from evaluatorq.simulation.utils.run_store import build_simulation_run as _build_simulation_run

runner = CliRunner()


@pytest.mark.parametrize('command', ['simulate', 'run', 'generate', 'upload-dataset'])
def test_command_help_shows_examples(command: str) -> None:
    """The four non-trivial commands carry a copy-paste Examples block in --help."""
    result = runner.invoke(app, [command, '--help'])
    assert result.exit_code == 0, result.output
    assert 'Examples' in result.output


@pytest.mark.parametrize('command', ['simulate', 'run', 'generate', 'upload-dataset'])
def test_epilog_examples_use_only_real_flags(command: str) -> None:
    """Drift guard: every ``eq sim <cmd> …`` example line must use only real flags of
    that <cmd> — so a rename can't leave a dead example behind. Example lines may
    reference a *different* subcommand (e.g. generate shows the simulate hop); each
    line is validated against the command it actually invokes."""
    import re

    import typer

    click_group = typer.main.get_command(app)
    subcommands = click_group.commands  # pyright: ignore[reportAttributeAccessIssue]

    def valid_flags(name: str) -> set[str]:
        flags: set[str] = set()
        for param in subcommands[name].params:
            flags.update(param.opts)
            flags.update(param.secondary_opts)
        return flags

    epilog = subcommands[command].epilog or ''
    checked = 0
    for line in epilog.splitlines():
        m = re.search(r'eq sim ([a-z-]+)\b(.*)', line)
        if not m or m.group(1) not in subcommands:
            continue
        invoked, rest = m.group(1), m.group(2)
        used = set(re.findall(r'(?<!\w)(--[a-z][a-z-]+|-[a-z])\b', rest))
        unknown = used - valid_flags(invoked)
        assert not unknown, f'{command} epilog line invokes `{invoked}` with unknown flags: {sorted(unknown)}'
        checked += 1
    assert checked, f'{command} epilog had no recognizable `eq sim <cmd>` example lines'


@pytest.mark.parametrize(
    ('command', 'long', 'short'),
    [
        ('generate', '--datapoints', '-d'),
        ('simulate', '--results', '-r'),
        ('run', '--datapoints', '-d'),
        ('run', '--results', '-r'),
    ],
)
def test_output_flags_expose_short_aliases(command: str, long: str, short: str) -> None:
    """Guard the self-describing output flags keep both their long and short spellings."""
    import typer

    subcommands = typer.main.get_command(app).commands  # pyright: ignore[reportAttributeAccessIssue]
    opts = {opt for param in subcommands[command].params for opt in param.opts}
    assert long in opts, f'{command} missing {long}'
    assert short in opts, f'{command} missing {short}'


def test_sim_help_describes_the_pipeline() -> None:
    result = runner.invoke(app, ['--help'])

    assert result.exit_code == 0, result.output
    assert 'generate' in result.output
    assert 'simulate' in result.output
    assert 'dashboard' in result.output


def test_provider_context_prefers_orq(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'orq-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-secret')

    _echo_using('openai/gpt-5.4-mini')

    output = capsys.readouterr().err
    assert output == 'Using for generations: Orq router · openai/gpt-5.4-mini\n'
    assert 'secret' not in output


def test_provider_context_uses_openai_when_orq_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-secret')

    _echo_using('gpt-4o-mini')

    output = capsys.readouterr().err
    assert output == 'Using for generations: OpenAI-compatible · gpt-4o-mini\n'
    assert 'secret' not in output


def test_shell_path_quotes_paths_with_spaces() -> None:
    assert _shell_path(Path('/tmp/sim runs')) == "'/tmp/sim runs'"


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
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import (
        SimulationResult,
        TerminatedBy,
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
        Persona,
        Scenario,
        SimulationDatapoint,
    )

    return [
        SimulationDatapoint(
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


def _stub_run(
    results: list[Any] | None = None,
    *,
    mode: str = "simulate",
    evaluator_names: list[str] | None = None,
    experiment_url: str | None = None,
) -> Any:
    """Build a real ``SimulationRun`` for mocking ``_simulate_impl``/``_run_impl``.

    The CLI now passes the run straight into real (unmocked) helpers like
    ``_auto_save_run``/``_write_report`` which need a genuine pydantic model,
    not a ``MagicMock``.
    """
    return _build_simulation_run(
        run_name="test-run",
        mode=mode,
        target_kind="openai_model",
        target="gpt-4o",
        target_model="gpt-4o",
        evaluator_names=evaluator_names or [],
        results=results if results is not None else [_make_result()],
        experiment_url=experiment_url,
    )


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_configure_logging_aligns_with_redteam_and_does_not_enable_http_client_info_logs() -> None:
    import logging

    root = logging.getLogger()
    previous_root_level = root.level
    previous_evaluatorq_level = logging.getLogger("evaluatorq").level
    previous_levels = {
        name: logging.getLogger(name).level for name in ("httpx", "httpcore", "openai")
    }
    root.setLevel(logging.INFO)
    for name in previous_levels:
        logging.getLogger(name).setLevel(logging.NOTSET)

    try:
        _configure_logging(0)
        assert logging.getLogger("evaluatorq").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.NOTSET
        assert logging.getLogger("httpcore").level == logging.NOTSET
        assert logging.getLogger("openai").level == logging.NOTSET

        _configure_logging(1)
        assert logging.getLogger("evaluatorq").level == logging.INFO

        _configure_logging(2)
        assert logging.getLogger("evaluatorq").level == logging.DEBUG

        _configure_logging(-1)
        assert logging.getLogger("evaluatorq").level == logging.ERROR
    finally:
        root.setLevel(previous_root_level)
        logging.getLogger("evaluatorq").setLevel(previous_evaluatorq_level)
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


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
# target resolution
# ---------------------------------------------------------------------------


def test_resolve_target_agent_prefix_uses_openresponses_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    target = _resolve_target(target="agent:refund-agent-fixed", vercel_url=None, openai_model=None)

    assert target.config.model == "agent/refund-agent-fixed"
    assert target.require_orq is True


def test_resolve_target_bare_value_defaults_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    target = _resolve_target(target="refund-agent-fixed", vercel_url=None, openai_model=None)

    assert target.config.model == "agent/refund-agent-fixed"


def test_resolve_target_deployment_prefix_uses_deployment_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = object()
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    with patch("evaluatorq.simulation.adapters.from_orq_deployment", return_value=marker) as factory:
        target = _resolve_target(target="deployment:refund-agent-fixed", vercel_url=None, openai_model=None)

    assert target is marker
    factory.assert_called_once_with("refund-agent-fixed")


@pytest.mark.asyncio
async def test_resolve_agent_description_fetches_agent_context_description() -> None:
    from evaluatorq.contracts import AgentContext

    backend = MagicMock()
    backend.resolve_context = AsyncMock(
        return_value=AgentContext(key="refund-agent-fixed", description="Handles refunds.")
    )
    with patch("evaluatorq.simulation.cli._make_sim_agent_backend", return_value=backend):
        description = await _resolve_agent_description(
            agent_description=None,
            target="agent:refund-agent-fixed",
        )

    assert description == "Handles refunds."
    backend.resolve_context.assert_awaited_once_with("refund-agent-fixed")


@pytest.mark.asyncio
async def test_resolve_agent_description_explicit_value_wins() -> None:
    description = await _resolve_agent_description(
        agent_description="Explicit bot description.",
        target="agent:refund-agent-fixed",
    )

    assert description == "Explicit bot description."


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


def test_validate_signposts_next_step(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path, count=2)
    result = runner.invoke(app, ["validate", "--input", str(dp_file)])
    assert result.exit_code == 0
    assert f"eq sim simulate -i {dp_file} --target agent:<your-agent-key>" in result.output


def test_validate_dataset_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-dataset", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0


def test_validate_dataset_bad_lines(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"not": "a datapoint"}\n', encoding="utf-8")
    result = runner.invoke(app, ["validate-dataset", str(bad)])
    assert result.exit_code == 1


def test_validate_accepts_input_option(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path, count=2)

    result = runner.invoke(app, ["validate", "--input", str(dp_file)])

    assert result.exit_code == 0, result.output
    assert "2 valid" in result.stdout


def test_validate_rejects_duplicate_input_paths(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)

    result = runner.invoke(app, ["validate", str(dp_file), "--input", str(dp_file)])

    assert result.exit_code != 0
    assert "once" in result.output


def test_simulate_accepts_input_option(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)

    with (
        patch("evaluatorq.simulation.cli._resolve_target", return_value=MagicMock()),
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock, return_value=_stub_run([], mode="simulate")),
    ):
        result = runner.invoke(
            app,
            ["simulate", "--input", str(dp_file), "--openai-model", "gpt-4o", "--no-save"],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output


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


def test_runs_suggests_dashboard_directory(tmp_path: Path) -> None:
    runs_dir = tmp_path / 'sim runs'
    runs_dir.mkdir()
    _write_run_file(runs_dir / 'my-run_20260101-000000.json', name='my-run')

    result = runner.invoke(app, ["runs", str(runs_dir)])

    assert result.exit_code == 0, result.output
    assert f"open: eq dashboard '{runs_dir}'" in result.stdout


def test_runs_full_does_not_truncate(tmp_path: Path) -> None:
    """--full renders at content width: the long filename and every scorer stay intact."""
    runs_dir = tmp_path / "sim-runs"
    runs_dir.mkdir()
    long_name = "my-run_20260713-220000.json"
    (runs_dir / long_name).write_text(
        json.dumps({
            "run_name": "sim",
            "created_at": "2026-07-13T22:00:00+00:00",
            "mode": "run",
            "target_kind": "orq_agent",
            "evaluator_names": ["goal_achieved", "criteria_met", "safety"],
            "total_results": 9,
            "scorer_averages": {"goal_achieved": 0.67, "criteria_met": 0.80, "safety": 0.55},
            "results": [],
        }),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["runs", str(runs_dir), "--full"])
    assert result.exit_code == 0, result.output
    # No ellipsis anywhere, filename contiguous, and the last (widest) scorer survives.
    assert "…" not in result.stdout
    assert long_name in result.stdout
    assert "safety=0.55" in result.stdout


def test_runs_skips_malformed(tmp_path: Path) -> None:
    runs_dir = tmp_path / "sim-runs"
    runs_dir.mkdir()
    (runs_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    result = runner.invoke(app, ["runs", str(runs_dir)])
    assert result.exit_code == 0
    assert "malformed" in result.output


# ---------------------------------------------------------------------------
# ui command  (Streamlit launcher)
# ---------------------------------------------------------------------------


def _write_run_file(path: Path, name: str = "r") -> None:
    path.write_text(
        json.dumps(
            {
                "run_name": name,
                "created_at": "2026-01-01T00:00:00+00:00",
                "mode": "run",
                "target_kind": "openai_model",
                "evaluator_names": [],
                "total_results": 0,
                "scorer_averages": {},
                "results": [],
            }
        ),
        encoding="utf-8",
    )


def test_ui_latest_resolves_most_recent(tmp_path: Path) -> None:
    runs_dir = tmp_path / "sim-runs"
    runs_dir.mkdir()
    older = runs_dir / "a_20260101-000000.json"
    newer = runs_dir / "b_20260102-000000.json"
    _write_run_file(older)
    _write_run_file(newer)
    import os

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    with (
        patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=runs_dir),
        patch("evaluatorq.common.ui.launch.launch_streamlit") as launch,
    ):
        result = runner.invoke(app, ["ui", "--latest"])

    assert result.exit_code == 0
    assert launch.call_args.args[1] == newer.resolve()


def test_ui_no_runs_errors(tmp_path: Path) -> None:
    with patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=tmp_path / "empty"):
        result = runner.invoke(app, ["ui"])
    assert result.exit_code == 1
    assert "No runs found" in result.output


def test_ui_explicit_path(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    _write_run_file(run_file)
    with (
        patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=tmp_path / "sim-runs"),
        patch("evaluatorq.common.ui.launch.launch_streamlit") as launch,
    ):
        result = runner.invoke(app, ["ui", str(run_file)])
    assert result.exit_code == 0
    assert launch.call_args.args[1] == run_file.resolve()
    assert 'deprecated' in result.stderr.lower()
    assert 'eq dashboard' in result.stderr


def test_ui_bare_filename_fallback(tmp_path: Path) -> None:
    runs_dir = tmp_path / "sim-runs"
    runs_dir.mkdir()
    run_file = runs_dir / "my-run.json"
    _write_run_file(run_file)
    with (
        patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=runs_dir),
        patch("evaluatorq.common.ui.launch.launch_streamlit") as launch,
    ):
        result = runner.invoke(app, ["ui", "my-run.json"])
    assert result.exit_code == 0
    assert launch.call_args.args[1] == run_file


def test_ui_missing_path_errors(tmp_path: Path) -> None:
    with patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=tmp_path / "sim-runs"):
        result = runner.invoke(app, ["ui", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


# ---------------------------------------------------------------------------
# simulate command — target validation  (datapoints in, no generation)
# ---------------------------------------------------------------------------


def test_simulate_requires_target(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(app, ["simulate", "--input", str(dp_file)])
    assert result.exit_code != 0


def test_simulate_rejects_multiple_targets(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "simulate",
            "--input", str(dp_file),
            "--target", "agent:k",
            "--agent-key", "k",
        ],
        env={"ORQ_API_KEY": "test-key"},
    )
    assert result.exit_code != 0


def test_simulate_missing_datapoints_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "simulate",
            "--input", str(tmp_path / "no.jsonl"),
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
            "--input", str(dp_file),
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
        mock_impl.return_value = _stub_run(results, mode="simulate")

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
                "--openai-model", "gpt-4o",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    # summary now renders via on_run_complete (mocked away here); covered by
    # tests/simulation/test_hooks.py and tests/simulation/reports/test_display.py.
    assert result.exit_code == 0


def test_simulate_saved_run_suggests_dashboard_directory(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    runs_dir = tmp_path / 'sim runs'
    saved_run = runs_dir / 'sim_20260713-220000.json'

    with (
        patch("evaluatorq.simulation.cli._resolve_target", return_value=MagicMock()),
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock, return_value=_stub_run([], mode="simulate")),
        patch("evaluatorq.simulation.cli._auto_save_run", return_value=saved_run),
        patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=runs_dir),
    ):
        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
                "--openai-model", "gpt-4o",
                "--yes",
                "--no-executive-summary",
            ],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    # Scope to the next-step CTA — the run filename legitimately appears in other
    # stderr lines (e.g. the "Run saved" log). The dashboard hand-off must point
    # at the directory, not the specific run file.
    handoff = result.stderr.split('▸ Next', 1)[1]
    assert "eq dashboard '" in handoff
    assert str(runs_dir) in handoff
    assert saved_run.name not in handoff


def test_run_saved_run_suggests_dashboard_directory(tmp_path: Path) -> None:
    runs_dir = tmp_path / 'sim-runs'
    saved_run = runs_dir / 'run_20260713-220000.json'

    with (
        patch("evaluatorq.simulation.cli._resolve_target", return_value=MagicMock()),
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock, return_value=_stub_run([], mode="run")),
        patch("evaluatorq.simulation.cli._auto_save_run", return_value=saved_run),
        patch("evaluatorq.simulation.cli._get_sim_runs_dir", return_value=runs_dir),
    ):
        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o",
                "--yes",
                "--no-executive-summary",
            ],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert f"eq dashboard {runs_dir}" in result.stderr


def test_simulate_writes_results_file(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    out_file = tmp_path / "out.jsonl"
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
        patch("evaluatorq.simulation.utils.dataset_export.export_results_to_jsonl") as mock_export,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="simulate")
        mock_export.return_value = None

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
                "--openai-model", "gpt-4o",
                "--results", str(out_file),
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mock_export.assert_called_once()


def test_simulate_report_writes_full_report(
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
        mock_impl.return_value = _stub_run([_make_result(scorer_scores={"goal_achieved": 1.0})], mode="simulate")

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
                "--openai-model", "gpt-4o",
                "--report", str(report),
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


def test_run_report_and_autosave_both_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without --no-save, --report writes the explicit file AND the
    # auto-save still lands under .evaluatorq/sim-runs/ (independent sinks).
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "report.json"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run([_make_result()], mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "bot",
                "--openai-model", "gpt-4o",
                "--report", str(report),
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
            "--input", str(dp_file),
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
        mock_impl.return_value = _stub_run([], mode="simulate")

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
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

    async def fake_simulate_run(**kwargs):
        captured.update(kwargs)
        return _stub_run([])

    monkeypatch.setattr("evaluatorq.simulation.api._simulate_run", fake_simulate_run)
    with patch("evaluatorq.simulation.cli._resolve_target") as mock_target:
        mock_target.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
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
        mock_impl.return_value = _stub_run([], mode="simulate")

        result = runner.invoke(
            app,
            ["simulate", "--input", str(dp_file), "--openai-model", "gpt-4o", "--no-save"],
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
        mock_impl.return_value = _stub_run([], mode="simulate")

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
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


def test_run_target_agent_uses_context_description_when_omitted(tmp_path: Path) -> None:
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch(
            "evaluatorq.simulation.cli._resolve_agent_description",
            new_callable=AsyncMock,
        ) as mock_description,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_description.return_value = "Context description"
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--target", "agent:refund-agent-fixed",
                "--no-save",
            ],
            env={"ORQ_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mock_description.assert_awaited_once_with(
        agent_description=None,
        target="agent:refund-agent-fixed",
    )
    assert mock_impl.call_args.kwargs["agent_description"] == "Context description"


def test_run_explicit_agent_description_does_not_resolve_context(tmp_path: Path) -> None:
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch(
            "evaluatorq.simulation.cli._resolve_agent_description",
            new_callable=AsyncMock,
        ) as mock_description,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_description.return_value = "Explicit description"
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--target", "agent:refund-agent-fixed",
                "--agent-description", "Explicit description",
                "--no-save",
            ],
            env={"ORQ_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mock_description.assert_awaited_once_with(
        agent_description="Explicit description",
        target="agent:refund-agent-fixed",
    )
    assert mock_impl.call_args.kwargs["agent_description"] == "Explicit description"


def test_run_requires_description_for_non_agent_target() -> None:
    with patch("evaluatorq.simulation.cli._resolve_target") as mock_target:
        result = runner.invoke(
            app,
            ["run", "--target", "deployment:refund-agent-fixed", "--no-save"],
            env={"ORQ_API_KEY": "test-key"},
        )

    assert result.exit_code == 1
    assert "requires --agent-description" in result.output
    mock_target.assert_not_called()


def test_simulate_invalid_target_prefix_is_clean(tmp_path: Path) -> None:
    dp_file = _make_datapoints_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "simulate",
            "--input", str(dp_file),
            "--target", "unknown:refund-agent-fixed",
            "--no-save",
        ],
        env={"ORQ_API_KEY": "test-key"},
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Unknown target kind" in result.output
    assert "Traceback" not in result.output


def test_run_success_no_save(tmp_path: Path) -> None:
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="run")

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
    # summary now renders via on_run_complete (mocked away here); covered by
    # tests/simulation/test_hooks.py and tests/simulation/reports/test_display.py.
    assert result.exit_code == 0


def test_run_forwards_flags(tmp_path: Path) -> None:
    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run([_make_result()], mode="run")

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
            ["simulate", "--input", str(dp_file), "--openai-model", "gpt-4o", "--no-save"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# --report-md / --report-html
# ---------------------------------------------------------------------------


def test_run_report_md_writes_dated_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--report-md writes exactly one *.md with non-empty content."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]
    export_dir = tmp_path / "exports"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o-mini",
                "--yes",
                "--no-save",
                "--report-md", str(export_dir),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mds = list(export_dir.glob("*.md"))
    assert len(mds) == 1, f"Expected 1 .md file, got {mds}"
    assert mds[0].read_text().strip(), "Markdown file should be non-empty"


def test_run_report_html_writes_dated_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--report-html writes exactly one *.html with non-empty content."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]
    export_dir = tmp_path / "exports"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o-mini",
                "--yes",
                "--no-save",
                "--report-html", str(export_dir),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    htmls = list(export_dir.glob("*.html"))
    assert len(htmls) == 1, f"Expected 1 .html file, got {htmls}"
    assert htmls[0].read_text().strip(), "HTML file should be non-empty"


def test_run_report_md_to_explicit_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--report-md with a file path (has a suffix) writes that exact file, not an auto-named one."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]
    out_file = tmp_path / "reports" / "my-report.md"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o-mini",
                "--yes",
                "--no-save",
                "--report-md", str(out_file),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert out_file.exists(), "explicit --report-md file path should be honoured verbatim"
    assert out_file.read_text().strip()
    # No auto-named sibling was created.
    assert list((tmp_path / "reports").glob("sim-report-*.md")) == []


def test_run_report_json_to_directory_autonames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--report with a directory drops an auto-named sim-report-*.json inside — same rule as --report-md/html."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]
    out_dir = tmp_path / "json-reports"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o-mini",
                "--yes",
                "--no-save",
                "--report", str(out_dir),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    jsons = list(out_dir.glob("sim-report-*.json"))
    assert len(jsons) == 1, f"Expected 1 auto-named .json, got {jsons}"
    assert jsons[0].read_text().strip()


def test_simulate_report_md_writes_dated_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """simulate --report-md writes exactly one *.md with non-empty content."""
    monkeypatch.chdir(tmp_path)
    dp_file = _make_datapoints_file(tmp_path)
    results = [_make_result(scorer_scores={"goal_achieved": 1.0})]
    export_dir = tmp_path / "exports"

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="simulate")

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
                "--openai-model", "gpt-4o-mini",
                "--yes",
                "--no-save",
                "--report-md", str(export_dir),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mds = list(export_dir.glob("*.md"))
    assert len(mds) == 1, f"Expected 1 .md file, got {mds}"
    assert mds[0].read_text().strip(), "Markdown file should be non-empty"


# ---------------------------------------------------------------------------
# generate command (datapoints only, no simulation)
# ---------------------------------------------------------------------------


def test_generate_requires_datapoints(tmp_path: Path) -> None:
    # --datapoints is required: gen-only must write the datapoints somewhere.
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
                "--datapoints", str(out_file),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "Generated 3 datapoint" in result.output
    assert out_file.exists()
    assert len([ln for ln in out_file.read_text().splitlines() if ln.strip()]) == 3


def test_generate_forwards_seed_flags(tmp_path: Path) -> None:
    """--persona-seed / --scenario-seed (repeatable) reach generate() as lists."""
    out_file = tmp_path / "dp.jsonl"

    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = _make_datapoints(2)

        result = runner.invoke(
            app,
            [
                "generate",
                "--agent-description", "A helpful bot",
                "--datapoints", str(out_file),
                "--persona-seed", "angry retiree",
                "--persona-seed", "fraud dispute",
                "--scenario-seed", "disputes a refund denial",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_impl.call_args.kwargs
    assert kwargs["persona_seeds"] == ["angry retiree", "fraud dispute"]
    assert kwargs["scenario_seeds"] == ["disputes a refund denial"]


def test_generate_no_seed_flags_pass_none(tmp_path: Path) -> None:
    """Without seed flags, generate() receives None (auto-generation path)."""
    out_file = tmp_path / "dp.jsonl"

    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = _make_datapoints(1)

        result = runner.invoke(
            app,
            ["generate", "--agent-description", "A helpful bot", "--datapoints", str(out_file)],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_impl.call_args.kwargs
    assert kwargs["persona_seeds"] is None
    assert kwargs["scenario_seeds"] is None


def test_generate_signposts_preview_and_next_step(tmp_path: Path) -> None:
    # generate must end by naming the next hop (the loudest seam) and preview
    # the personas/scenarios it built.
    out_file = tmp_path / "dp.jsonl"
    datapoints = _make_datapoints(2)

    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = datapoints

        result = runner.invoke(
            app,
            [
                "generate",
                "--target", "agent:refund-agent-fixed",
                "--agent-description", "A helpful bot",
                "--datapoints", str(out_file),
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "Personas" in result.output and "User0" in result.output
    assert "Scenarios" in result.output
    assert f"eq sim simulate -i {out_file} --target agent:refund-agent-fixed" in result.output


def test_generate_quiet_suppresses_preview_and_next_step(tmp_path: Path) -> None:
    # --quiet is for scripts/pipes: suppress the plan box, preview, and next-step
    # CTA (all non-error signposting). Keep only the one-line result confirmation.
    out_file = tmp_path / "dp.jsonl"
    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = _make_datapoints(2)
        result = runner.invoke(
            app,
            ["generate", "--agent-description", "A bot", "--datapoints", str(out_file), "--quiet"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert "Personas" not in result.output
    assert "Generate Plan" not in result.output
    assert "▸ Next" not in result.output
    assert "eq sim simulate" not in result.output
    # the one-line result is still shown
    assert "Generated" in result.output and "datapoint(s)" in result.output


def test_generate_target_agent_uses_context_description_when_omitted(tmp_path: Path) -> None:
    out_file = tmp_path / "dp.jsonl"
    datapoints = _make_datapoints(1)

    with (
        patch(
            "evaluatorq.simulation.cli._resolve_agent_description",
            new_callable=AsyncMock,
        ) as mock_description,
        patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_description.return_value = "Context description"
        mock_impl.return_value = datapoints

        result = runner.invoke(
            app,
            [
                "generate",
                "--target", "agent:refund-agent-fixed",
                "--datapoints", str(out_file),
            ],
            env={"ORQ_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    mock_description.assert_awaited_once_with(
        agent_description=None,
        target="agent:refund-agent-fixed",
    )
    assert mock_impl.call_args.kwargs["agent_description"] == "Context description"


def test_generate_datapoints_roundtrips_through_simulate_loader(tmp_path: Path) -> None:
    # The whole point of gen-only: the file `generate` writes must load back
    # via the same loader `simulate --datapoints` uses, with id + fields intact.
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl

    out_file = tmp_path / "dp.jsonl"
    datapoints = _make_datapoints(2)

    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = datapoints
        result = runner.invoke(
            app,
            ["generate", "--agent-description", "bot", "--datapoints", str(out_file)],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    # The Generate Plan box surfaces the resolved provider + model (beats 1+2).
    assert 'Generate Plan' in result.stderr
    assert 'OpenAI-compatible' in result.stderr and 'openai/gpt-5.4-mini' in result.stderr
    loaded = load_datapoints_from_jsonl(str(out_file))
    assert [dp.id for dp in loaded] == ["dp-0", "dp-1"]  # id round-trips (not re-fabricated)
    assert [dp.persona.name for dp in loaded] == ["User0", "User1"]
    assert [dp.scenario.name for dp in loaded] == ["Scenario0", "Scenario1"]
    assert all(dp.first_message == "Hello" for dp in loaded)


def test_generate_datapoints_passes_validate_dataset(tmp_path: Path) -> None:
    # generate's output must validate under the tool's own validate-dataset.
    out_file = tmp_path / "dp.jsonl"
    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.return_value = _make_datapoints(2)
        gen = runner.invoke(
            app,
            ["generate", "--agent-description", "bot", "--datapoints", str(out_file)],
            env={"OPENAI_API_KEY": "test-key"},
        )
    assert gen.exit_code == 0, gen.output

    validated = runner.invoke(app, ["validate-dataset", str(out_file)])
    assert validated.exit_code == 0, validated.output
    assert "2 valid datapoint" in validated.output


def test_generate_no_datapoints_runtime_error_is_clean(tmp_path: Path) -> None:
    # RuntimeError from generation (e.g. every persona x scenario pair failed)
    # surfaces as a one-line error, not a traceback.
    out_file = tmp_path / "dp.jsonl"
    with patch("evaluatorq.simulation.cli._generate_impl", new_callable=AsyncMock) as mock_impl:
        mock_impl.side_effect = RuntimeError("first-message generation produced no datapoints")
        result = runner.invoke(
            app,
            ["generate", "--agent-description", "bot", "--datapoints", str(out_file)],
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
            "--datapoints", str(out_file),
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
                "--datapoints", str(out_file),
                "--sim-model", "custom-model",
                "--num-personas", "2",
                "--num-scenarios", "4",
            ],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
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

    async def fake_generate_and_simulate_run(**kwargs):
        captured.update(kwargs)
        return _stub_run([], mode="run")

    monkeypatch.setattr(
        "evaluatorq.simulation.api._generate_and_simulate_run", fake_generate_and_simulate_run
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
            "--datapoints", "dp.jsonl",
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
# run --datapoints  (Task 3)
# ---------------------------------------------------------------------------


def test_run_datapoints_writes_inputs(tmp_path: Path) -> None:
    """--datapoints writes the simulate inputs as JSONL and echoes a status message."""
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl

    dp_file = tmp_path / "dp.jsonl"

    async def fake_generate_and_simulate_run(**kwargs: Any) -> Any:
        emit = kwargs.get("emit_datapoints")
        if emit is not None:
            emit(_make_datapoints(2))
        return _stub_run([], mode="run")

    with (
        patch(
            "evaluatorq.simulation.api._generate_and_simulate_run",
            side_effect=fake_generate_and_simulate_run,
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
                "--datapoints", str(dp_file),
                "--no-save",
            ],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert dp_file.exists(), "Datapoints file was not created"
    loaded = load_datapoints_from_jsonl(str(dp_file))
    assert len(loaded) == 2
    assert [dp.id for dp in loaded] == ["dp-0", "dp-1"]
    # Status message echoed to stderr (mix_stderr=True is typer runner default)
    assert "Datapoints saved" in result.output


def test_run_datapoints_echoes_even_when_simulation_fails(tmp_path: Path) -> None:
    """The save confirmation must survive a later simulation failure.

    Datapoints are written (and emit_datapoints called) before simulation runs,
    so a RuntimeError from simulation must not swallow the "Datapoints saved"
    message — those frozen inputs are exactly what you re-feed to `sim simulate`.
    """
    dp_file = tmp_path / "dp.jsonl"

    async def fake_generate_and_simulate_run(**kwargs: Any) -> Any:
        emit = kwargs.get("emit_datapoints")
        if emit is not None:
            emit(_make_datapoints(2))
        raise RuntimeError("simulation produced no datapoints")

    with (
        patch(
            "evaluatorq.simulation.api._generate_and_simulate_run",
            side_effect=fake_generate_and_simulate_run,
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
                "--datapoints", str(dp_file),
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 1, result.output
    assert dp_file.exists(), "Datapoints file should be written before sim runs"
    assert "Datapoints saved" in result.output
    assert "Error: simulation produced no datapoints" in result.output


def test_run_without_datapoints_writes_no_file(tmp_path: Path) -> None:
    """When --datapoints is omitted, emit_datapoints=None is passed and no file is created."""
    captured_emit: dict[str, Any] = {}

    async def fake_generate_and_simulate_run(**kwargs: Any) -> Any:
        captured_emit["emit"] = kwargs.get("emit_datapoints")
        return _stub_run([], mode="run")

    with (
        patch(
            "evaluatorq.simulation.api._generate_and_simulate_run",
            side_effect=fake_generate_and_simulate_run,
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


# ---------------------------------------------------------------------------
# --yes / -y flag wiring tests (Task 7)
# ---------------------------------------------------------------------------


def test_simulate_yes_exits_clean(tmp_path: Path) -> None:
    """--yes flag is accepted and wired correctly; CLI exits 0 with mocked impl."""
    dp_file = _make_datapoints_file(tmp_path)
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._simulate_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="simulate")

        result = runner.invoke(
            app,
            [
                "simulate",
                "--input", str(dp_file),
                "--openai-model", "gpt-4o",
                "--yes",
                "--no-save",
            ],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert 'Using for generations: OpenAI-compatible · openai/gpt-5.4-mini' in result.stderr


def test_run_yes_exits_clean(tmp_path: Path) -> None:
    """--yes flag is accepted and wired correctly; CLI exits 0 with mocked impl."""
    results = [_make_result()]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run(results, mode="run")

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "A helpful bot",
                "--openai-model", "gpt-4o",
                "--yes",
                "--no-save",
            ],
            env={"ORQ_API_KEY": "", "OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert 'Using for generations: OpenAI-compatible · openai/gpt-5.4-mini' in result.stderr


def test_export_md_includes_stored_recommendations(tmp_path):
    """eq sim export --format md renders a run JSON's stored suggestions."""
    import json as _json

    from evaluatorq.simulation.types import (
        SimulationRecommendation,
        SimulationResult,
        SimulationRun,
        TerminatedBy,
    )
    from evaluatorq.contracts import Message, TokenUsage

    result = SimulationResult(
        messages=[Message(role="user", content="hi"), Message(role="assistant", content="yo")],
        terminated_by=TerminatedBy.judge,
        reason="done",
        goal_achieved=False,
        goal_completion_score=0.1,
        rules_broken=[],
        turn_count=1,
        turn_metrics=[],
        token_usage=TokenUsage(total_tokens=10),
        metadata={"persona": "P", "scenario": "S"},
    )
    from datetime import datetime, timezone

    run = SimulationRun(
        run_name="t",
        created_at=datetime.now(tz=timezone.utc),
        mode="run",
        target_kind="orq_agent",
        evaluator_names=[],
        total_results=1,
        scorer_averages={},
        results=[result],
        recommendations=[
            SimulationRecommendation(
                result_index=0,
                datapoint_id="dp-1",
                persona="P",
                scenario="S",
                triggers=["criterion_failed: says hello politely"],
                suggestions=["Add a greeting instruction to the system prompt."],
            )
        ],
    )
    src = tmp_path / "run.json"
    src.write_text(run.model_dump_json())
    out = tmp_path / "report.md"

    runner = CliRunner()
    res = runner.invoke(app, ["export", "-i", str(src), "-o", str(out), "--format", "md"])
    assert res.exit_code == 0, res.output
    md = out.read_text()
    assert "Remediation Suggestions" in md
    assert "Add a greeting instruction to the system prompt." in md
    assert "Criterion failed: says hello politely" in md


def test_export_html_format(tmp_path):
    """eq sim export --format html writes a self-contained HTML report."""
    results = tmp_path / "results.jsonl"
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy
    from evaluatorq.contracts import Message, TokenUsage

    r = SimulationResult(
        messages=[Message(role="user", content="hi")],
        terminated_by=TerminatedBy.judge,
        reason="done",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        turn_metrics=[],
        token_usage=TokenUsage(total_tokens=10),
        metadata={"persona": "P", "scenario": "S"},
    )
    results.write_text(r.model_dump_json() + "\n")
    out = tmp_path / "report.html"

    runner = CliRunner()
    res = runner.invoke(app, ["export", "-i", str(results), "-o", str(out), "--format", "html"])
    assert res.exit_code == 0, res.output
    assert "<!DOCTYPE html>" in out.read_text()


def test_export_rejects_unknown_format(tmp_path):
    src = tmp_path / "x.jsonl"
    src.write_text("")
    runner = CliRunner()
    res = runner.invoke(app, ["export", "-i", str(src), "-o", str(tmp_path / "y"), "--format", "pdf"])
    assert res.exit_code != 0


def test_run_recommendations_flag_attaches_to_saved_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eq sim run --recommendations generates suggestions and stores them on the run report."""
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "report.json"
    fake_recs = [
        {
            "result_index": 0,
            "datapoint_id": "dp-1",
            "persona": "P",
            "scenario": "S",
            "triggers": ["Criterion failed: x"],
            "suggestions": ["Fix x."],
        }
    ]

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        from evaluatorq.simulation.types import SimulationRecommendation

        stub_run = _stub_run([_make_result()], mode="run")
        stub_run.recommendations = [SimulationRecommendation.model_validate(r) for r in fake_recs]
        mock_impl.return_value = stub_run

        result = runner.invoke(
            app,
            [
                "run",
                "--agent-description", "bot",
                "--openai-model", "gpt-4o",
                "--report", str(report),
                "--recommendations",
                "--no-save",
            ],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert mock_impl.await_args.kwargs['recommendations'] is True
    saved = json.loads(report.read_text())
    assert saved["recommendations"][0]["suggestions"] == ["Fix x."]


def test_run_without_recommendations_flag_skips_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with (
        patch("evaluatorq.simulation.cli._resolve_target") as mock_target,
        patch("evaluatorq.simulation.cli._run_impl", new_callable=AsyncMock) as mock_impl,
    ):
        mock_target.return_value = MagicMock()
        mock_impl.return_value = _stub_run([_make_result()], mode="run")

        result = runner.invoke(
            app,
            ["run", "--agent-description", "bot", "--openai-model", "gpt-4o", "--no-save"],
            env={"OPENAI_API_KEY": "test-key"},
        )

    assert result.exit_code == 0, result.output
    assert mock_impl.await_args.kwargs['recommendations'] is False
