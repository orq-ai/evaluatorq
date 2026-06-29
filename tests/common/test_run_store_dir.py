from __future__ import annotations

from pathlib import Path

import pytest

from evaluatorq.common.run_store_dir import get_store_dir


def test_env_var_override_is_used_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALUATORQ_DIR", str(tmp_path / "store"))
    assert get_store_dir("runs") == tmp_path / "store" / "runs"


def test_falls_back_to_cwd_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EVALUATORQ_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_store_dir("sim-runs") == tmp_path / ".evaluatorq" / "sim-runs"


def test_empty_env_var_treated_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALUATORQ_DIR", "")
    monkeypatch.chdir(tmp_path)
    assert get_store_dir("runs") == tmp_path / ".evaluatorq" / "runs"
