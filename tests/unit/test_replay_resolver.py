"""Resolution of a "previous run" reference to a stored run file."""
# ruff: noqa: S101

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from evaluatorq.common.replay import ReplayError, load_run_payload, resolve_run_path
from evaluatorq.common.run_manifest import start_manifest

if TYPE_CHECKING:
    from pathlib import Path


def _write_run(runs_dir: Path, name: str, *, payload: dict[str, object] | None = None, mtime: int | None = None) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f'{name}.json'
    path.write_text(json.dumps(payload if payload is not None else {'run_name': name}), encoding='utf-8')
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_latest_picks_the_newest_report(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _write_run(runs, 'old_20260101_000000', mtime=1_000_000)
    newest = _write_run(runs, 'new_20260102_000000', mtime=2_000_000)

    assert resolve_run_path('latest', runs, surface='red team') == newest


def test_latest_without_runs_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match='nothing to replay'):
        resolve_run_path('latest', tmp_path / 'runs', surface='red team')


def test_file_name_with_and_without_extension(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    path = _write_run(runs, 'rt_20260101_000000')

    assert resolve_run_path('rt_20260101_000000.json', runs, surface='red team') == path
    assert resolve_run_path('rt_20260101_000000', runs, surface='red team') == path


def test_explicit_path_outside_the_runs_dir(tmp_path: Path) -> None:
    elsewhere = _write_run(tmp_path / 'elsewhere', 'saved')

    assert resolve_run_path(str(elsewhere), tmp_path / 'runs', surface='red team') == elsewhere


def test_run_name_prefix_resolves_to_newest_matching_run(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _write_run(runs, 'nightly_20260101_000000', mtime=1_000_000)
    newest = _write_run(runs, 'nightly_20260202_000000', mtime=2_000_000)

    assert resolve_run_path('nightly', runs, surface='red team') == newest


def test_manifest_run_id_and_unambiguous_prefix(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    report = _write_run(runs, 'rt_20260101_000000')
    writer = start_manifest(run_id='a1b2c3d4e5f6', surface='redteam', run_name='rt', runs_dir=runs)
    writer.complete(report_path=report)

    assert resolve_run_path('a1b2c3d4e5f6', runs, surface='red team') == report
    assert resolve_run_path('a1b2c3d4', runs, surface='red team') == report


def test_short_prefix_is_not_treated_as_a_run_id(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    report = _write_run(runs, 'rt_20260101_000000')
    start_manifest(run_id='a1b2c3d4e5f6', surface='redteam', run_name='rt', runs_dir=runs).complete(report_path=report)

    with pytest.raises(ReplayError, match='Could not resolve'):
        resolve_run_path('a1b2', runs, surface='red team')


def test_ambiguous_run_id_prefix_is_rejected(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    first = _write_run(runs, 'one_20260101_000000')
    second = _write_run(runs, 'two_20260101_000000')
    start_manifest(run_id='abcdefaa1111', surface='redteam', run_name='one', runs_dir=runs).complete(report_path=first)
    start_manifest(run_id='abcdefbb2222', surface='redteam', run_name='two', runs_dir=runs).complete(report_path=second)

    with pytest.raises(ReplayError, match='ambiguous'):
        resolve_run_path('abcdef', runs, surface='red team')


def test_unknown_reference_lists_recent_runs(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    _write_run(runs, 'rt_20260101_000000')

    with pytest.raises(ReplayError, match='rt_20260101_000000.json'):
        resolve_run_path('nope', runs, surface='red team')


def test_empty_reference_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match='Empty previous'):
        resolve_run_path('   ', tmp_path, surface='red team')


def test_load_run_payload_returns_dict_and_path(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    path = _write_run(runs, 'rt_20260101_000000', payload={'run_name': 'rt', 'datapoints': []})

    payload, resolved = load_run_payload('latest', runs, surface='red team')
    assert resolved == path
    assert payload['run_name'] == 'rt'


def test_load_run_payload_rejects_malformed_json(tmp_path: Path) -> None:
    runs = tmp_path / 'runs'
    runs.mkdir(parents=True)
    (runs / 'broken.json').write_text('{not json', encoding='utf-8')

    with pytest.raises(ReplayError, match='not valid JSON'):
        load_run_payload('broken', runs, surface='red team')
