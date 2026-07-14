"""Unit tests for the two untested error paths in
``evaluatorq.simulation.utils.run_store.auto_save_run``:

1. Collision exhaustion — every exclusive-create attempt (up to 1000) raises
   ``FileExistsError``, and the function ultimately raises ``RuntimeError``.
2. Orphan cleanup — the exclusive-create succeeds (file exists on disk) but
   the subsequent write raises ``OSError``; the partial file must be
   unlinked before the ``OSError`` is re-raised.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluatorq.simulation.utils.run_store import auto_save_run, build_simulation_run

# ---------------------------------------------------------------------------
# Helpers (mirrors tests/simulation/test_run_store.py::_make_result)
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


def _make_run(run_name: str = "collide") -> Any:
    return build_simulation_run(
        run_name=run_name,
        mode="run",
        target_kind="openai_model",
        evaluator_names=[],
        results=[_make_result()],
    )


# ---------------------------------------------------------------------------
# Collision exhaustion -> RuntimeError
# ---------------------------------------------------------------------------


def test_auto_save_run_raises_runtime_error_after_1000_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "sim-runs"
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: runs_dir,
    )

    call_count = 0

    def always_collide(self: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        raise FileExistsError(17, "File exists", str(self))

    monkeypatch.setattr(Path, "open", always_collide)

    run = _make_run("exhausted")

    with pytest.raises(RuntimeError, match=r"Could not find a free run-store filename.*1000 attempts"):
        auto_save_run(run=run, run_name="exhausted")

    # The loop runs `for counter in range(1000)`, so exactly 1000 open attempts.
    assert call_count == 1000
    # No files should have been left behind — every attempt raised before creation.
    assert not runs_dir.exists() or list(runs_dir.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Orphan cleanup on OSError during write
# ---------------------------------------------------------------------------


def test_auto_save_run_unlinks_partial_file_on_write_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "sim-runs"
    monkeypatch.setattr(
        "evaluatorq.simulation.utils.run_store.get_sim_runs_dir",
        lambda: runs_dir,
    )

    original_open = Path.open
    created_paths: list[Path] = []

    def open_then_fail_write(self: Path, *args: Any, **kwargs: Any) -> Any:
        # Let the real exclusive-create happen so the file actually lands on
        # disk (mirrors "disk full after create" scenarios), then sabotage
        # the returned handle's write() to raise OSError.
        fh = original_open(self, *args, **kwargs)
        created_paths.append(self)

        def bad_write(_data: str) -> int:
            raise OSError("disk full")

        fh.write = bad_write  # type: ignore[method-assign]
        return fh

    monkeypatch.setattr(Path, "open", open_then_fail_write)

    run = _make_run("orphan")

    with pytest.raises(OSError, match="disk full"):
        auto_save_run(run=run, run_name="orphan")

    assert len(created_paths) == 1
    # The partial file must have been unlinked by the except-OSError branch.
    assert not created_paths[0].exists()
    assert list(runs_dir.glob("*.json")) == []
