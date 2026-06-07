"""Smoke tests for `python -m evaluatorq` invocation.

The package ships console scripts (`eq`, `evaluatorq`) plus a
`python -m evaluatorq.cli` path. These tests assert the conventional
`python -m evaluatorq` entry point (via `__main__.py`) works too, so
tooling that probes `import evaluatorq` then falls back to
`python -m evaluatorq` does not break.
"""

from __future__ import annotations

import subprocess
import sys


def test_python_m_evaluatorq_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "evaluatorq", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Evaluation framework for AI systems." in result.stdout


def test_python_m_evaluatorq_cli_help_exits_zero() -> None:
    """The legacy `python -m evaluatorq.cli` path must keep working."""
    result = subprocess.run(
        [sys.executable, "-m", "evaluatorq.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
