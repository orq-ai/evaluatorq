"""CLI wiring for `eq sim simulate --from-run` (RES-1126)."""
# ruff: noqa: S101

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from evaluatorq.simulation.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _output(result) -> str:
    """Combined stdout+stderr, unwrapped.

    Typer renders usage errors through rich, which hard-wraps them into a box
    at the terminal width — so a substring assertion passes on a wide dev
    terminal and fails on a narrow CI one. Strip ANSI and the box borders, then
    collapse whitespace, so assertions are about the message not the viewport.
    """
    raw = result.output + (result.stderr or '')
    plain = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw)
    return re.sub(r'\s+', ' ', re.sub(r'[│╭╮╰╯─╷╵]', ' ', plain))


def test_from_run_is_forwarded_as_previous_run(monkeypatch) -> None:
    from evaluatorq.simulation.utils.run_store import build_simulation_run

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='orq_agent',
        evaluator_names=[],
        results=[],
    )
    with patch('evaluatorq.simulation.cli._simulate_impl', new=AsyncMock(return_value=run)) as impl:
        result = runner.invoke(
            app,
            ['simulate', '--from-run', 'latest', '--target', 'agent:demo', '--no-save', '--yes'],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, _output(result)
    assert impl.call_args.kwargs['previous_run'] == 'latest'


def test_input_and_from_run_are_mutually_exclusive(tmp_path: Path) -> None:
    dp = tmp_path / 'dp.jsonl'
    dp.write_text('', encoding='utf-8')

    result = runner.invoke(
        app,
        ['simulate', '--input', str(dp), '--from-run', 'latest', '--target', 'agent:demo', '--yes'],
    )

    assert result.exit_code != 0
    assert 'exactly one of --input, --dataset-id, or --from-run' in _output(result)


def test_no_input_source_is_rejected() -> None:
    result = runner.invoke(app, ['simulate', '--target', 'agent:demo', '--yes'])

    assert result.exit_code != 0
    assert 'exactly one of --input, --dataset-id, or --from-run' in _output(result)


def test_unresolvable_from_run_exits_cleanly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    result = runner.invoke(
        app,
        ['simulate', '--from-run', 'nope', '--target', 'agent:demo', '--no-save', '--yes'],
    )

    assert result.exit_code == 1
    assert 'Could not resolve previous simulation run' in _output(result)
