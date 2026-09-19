"""Tests for the input-source validation `eq sim simulate` performs before it runs anything.

These four branches live in `_resolve_simulate_options` and every one of them
raises `typer.BadParameter` before a target is resolved or a model is called,
so no fixture beyond `CliRunner` and `tmp_path` is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from evaluatorq.simulation.cli import app


runner = CliRunner()

_BAD_PARAMETER_EXIT_CODE = 2

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _squeezed(text: str) -> str:
    """Drop ANSI colour codes, rich's box glyphs, and all whitespace.

    Typer forces rich's colourised rendering whenever ``GITHUB_ACTIONS`` is set
    (see ``typer.rich_utils.FORCE_TERMINAL``), which CI always sets and a local
    shell normally does not. That highlighting splits option tokens like
    ``--input`` into separately-coloured runs (e.g. ``\\x1b[1;36m-\\x1b[0m\\x1b[1;36m-input\\x1b[0m``),
    so the escape codes must be stripped before whitespace, or a CI-only run
    leaves stray control sequences inside the squeezed string and breaks the match.
    """
    return ''.join(_ANSI_RE.sub('', text).replace('│', '').split())


def _assert_rejected(result: Result, expected: str) -> None:
    assert result.exit_code == _BAD_PARAMETER_EXIT_CODE, result.output
    assert _squeezed(expected) in _squeezed(result.output)


def test_rejects_zero_input_sources() -> None:
    result = runner.invoke(app, ['simulate', '--no-executive-summary'])

    _assert_rejected(result, 'Provide exactly one of --input, --dataset-id, --experiment-id, or --from-run.')


def test_rejects_two_input_sources(tmp_path: Path) -> None:
    datapoints = tmp_path / 'datapoints.jsonl'
    datapoints.write_text('', encoding='utf-8')

    result = runner.invoke(
        app,
        ['simulate', '--no-executive-summary', '--input', str(datapoints), '--dataset-id', 'ds-1'],
    )

    _assert_rejected(
        result,
        'Provide exactly one of --input, --dataset-id, --experiment-id, or --from-run '
        '(got: --input, --dataset-id).',
    )


def test_rejects_experiment_run_id_without_experiment_id(tmp_path: Path) -> None:
    datapoints = tmp_path / 'datapoints.jsonl'
    datapoints.write_text('', encoding='utf-8')

    result = runner.invoke(
        app,
        [
            'simulate',
            '--no-executive-summary',
            '--input',
            str(datapoints),
            '--experiment-run-id',
            'run-1',
        ],
    )

    _assert_rejected(result, '--experiment-run-id requires --experiment-id.')


def test_rejects_missing_datapoints_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    missing = tmp_path / 'absent.jsonl'

    result = runner.invoke(app, ['simulate', '--no-executive-summary', '--input', str(missing)])

    _assert_rejected(result, f'Datapoints file not found: {missing}')
