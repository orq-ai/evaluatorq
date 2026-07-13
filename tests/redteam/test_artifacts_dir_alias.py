"""Tests for the ``artifacts_dir``/``output_dir`` deprecation shim in red_team()."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from evaluatorq.redteam.runner import _resolve_artifacts_dir


def test_artifacts_dir_only_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = _resolve_artifacts_dir(artifacts_dir=Path('artifacts'), output_dir=None)

    assert result == Path('artifacts')


def test_output_dir_only_warns_and_returns_output_dir() -> None:
    with pytest.warns(DeprecationWarning, match='output_dir'):
        result = _resolve_artifacts_dir(artifacts_dir=None, output_dir=Path('legacy'))

    assert result == Path('legacy')


def test_both_passed_artifacts_dir_wins_but_still_warns() -> None:
    with pytest.warns(DeprecationWarning, match='output_dir'):
        result = _resolve_artifacts_dir(artifacts_dir=Path('new'), output_dir=Path('legacy'))

    assert result == Path('new')
