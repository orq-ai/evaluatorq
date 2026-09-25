from __future__ import annotations

from pathlib import Path

from evaluatorq.insights.store import list_runs, load_run, save_run


def test_round_trip_and_truncated_file(minimal_run, tmp_path: Path) -> None:
    path = save_run(minimal_run, tmp_path)
    assert path.name.startswith('insights_20260901-000000_minimal-run')
    assert load_run(path) == minimal_run

    broken = tmp_path / 'insights_20260902-broken.json'
    broken.write_text('{truncated', encoding='utf-8')
    listed = list_runs(tmp_path)
    assert [p for p, _ in listed] == [broken, path]
    assert isinstance(listed[0][1], str)
    assert isinstance(listed[1][1], type(minimal_run))


def test_same_name_and_second_never_overwrites(tmp_path: Path, minimal_run) -> None:
    first = save_run(minimal_run, tmp_path)
    second_run = minimal_run.model_copy(update={'run_id': 'run-2'})
    second = save_run(second_run, tmp_path)

    assert first != second
    assert first.name == 'insights_20260901-000000_minimal-run.json'
    assert second.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(first).run_id == 'run-1'
    assert load_run(second).run_id == 'run-2'
