"""Unit tests for the shared Streamlit launcher."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from evaluatorq.common.ui import launch as launch_mod
from evaluatorq.common.ui.launch import ensure_streamlit, launch_streamlit


def _script_and_report(tmp_path: Path) -> tuple[Path, Path]:
    script = tmp_path / "dash.py"
    script.write_text("", encoding="utf-8")
    report = tmp_path / "run.json"
    report.write_text("{}", encoding="utf-8")
    return script, report


def test_launch_streamlit_builds_command(tmp_path: Path) -> None:
    script, report = _script_and_report(tmp_path)
    with (
        patch.object(launch_mod.subprocess, "run", return_value=MagicMock(returncode=0)) as run,
        patch.object(launch_mod, "ensure_streamlit"),
        pytest.raises(typer.Exit) as exc,
    ):
        launch_streamlit(script, report, port=9000, host="0.0.0.0", extra="redteam")

    assert exc.value.exit_code == 0
    cmd = run.call_args.args[0]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert str(script) in cmd
    assert cmd[cmd.index("--server.port") + 1] == "9000"
    assert cmd[cmd.index("--server.address") + 1] == "0.0.0.0"
    # The report path is passed after the bare -- so Streamlit forwards it to argv.
    assert cmd[cmd.index("--") + 1] == str(report)


def test_launch_streamlit_propagates_exit_code(tmp_path: Path) -> None:
    script, report = _script_and_report(tmp_path)
    with (
        patch.object(launch_mod.subprocess, "run", return_value=MagicMock(returncode=3)),
        patch.object(launch_mod, "ensure_streamlit"),
        pytest.raises(typer.Exit) as exc,
    ):
        launch_streamlit(script, report, extra="redteam")
    assert exc.value.exit_code == 3


def test_launch_streamlit_missing_script(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as exc:
        launch_streamlit(tmp_path / "nope.py", tmp_path / "run.json", extra="redteam")
    assert exc.value.exit_code == 1


def test_ensure_streamlit_hint_names_extra(capsys: pytest.CaptureFixture[str]) -> None:
    # Setting the module to None in sys.modules makes `import streamlit` raise.
    with patch.dict(sys.modules, {"streamlit": None}), pytest.raises(typer.Exit) as exc:
        ensure_streamlit(extra="simulation")
    assert exc.value.exit_code == 1
    assert "evaluatorq[simulation]" in capsys.readouterr().err
