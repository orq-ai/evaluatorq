from __future__ import annotations

# ruff: noqa: S101
import errno
import os
from pathlib import Path

import pytest

from evaluatorq.insights import store
from evaluatorq.insights.models import LabelAnswer, LabelResult
from evaluatorq.insights.presets import FAILURE_TAXONOMY
from evaluatorq.insights.store import list_run_paths, list_runs, load_run, save_run


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


def test_saved_legacy_failure_taxonomy_warns_without_changing_saved_data(minimal_run, tmp_path: Path) -> None:
    import json

    assert isinstance(FAILURE_TAXONOMY.criteria, dict)
    legacy_criteria = FAILURE_TAXONOMY.criteria.copy()
    legacy_criteria.pop('No failure')
    legacy_spec = FAILURE_TAXONOMY.model_copy(update={
        'instructions': 'Classify the primary way the assistant failed in this trace.',
        'criteria': legacy_criteria,
    })
    traces = [trace.model_copy(update={'labels': {
        'failure_taxonomy': LabelAnswer(value='Hallucination', confidence=0.9, probabilities=None, error=None),
    }}) for trace in minimal_run.traces]
    result = LabelResult(
        spec=legacy_spec,
        counts={'Hallucination': len(traces)},
        mean_confidence=0.9,
        n_low_confidence=0,
        n_failed=0,
    )
    run = minimal_run.model_copy(update={
        'config': minimal_run.config.model_copy(update={'labels': [legacy_spec]}),
        'traces': traces,
        'labels': {'failure_taxonomy': result},
    })

    path = save_run(run, tmp_path)
    saved_bytes = path.read_bytes()
    restored = load_run(path)
    saved_model = restored.model_dump(mode='json')

    assert restored.config.labels == [legacy_spec]
    assert restored.labels['failure_taxonomy'].spec == legacy_spec
    assert [trace.labels['failure_taxonomy'].value for trace in restored.traces] == ['Hallucination'] * len(traces)
    assert restored.labels['failure_taxonomy'].counts == {'Hallucination': len(traces)}
    assert restored.warnings == run.warnings
    counts_warning = restored.labels['failure_taxonomy'].counts_warning
    assert counts_warning is not None
    assert 'may include clean traces' in counts_warning
    assert path.read_bytes() == saved_bytes
    assert json.loads(path.read_bytes()) == saved_model
    assert restored.model_dump(mode='json') == saved_model

    revised_criteria = {**legacy_criteria, 'No failure': 'The assistant did not fail in this trace.'}
    revised_spec = legacy_spec.model_copy(update={
        'instructions': FAILURE_TAXONOMY.instructions,
        'criteria': revised_criteria,
    })
    revised_trace = traces[0].model_copy(update={'labels': {
        **traces[0].labels,
        'failure_taxonomy': LabelAnswer(value='No failure', confidence=0.9, probabilities=None, error=None),
    }})
    revised = LabelResult(
        spec=revised_spec,
        counts={'No failure': 1},
        mean_confidence=0.9,
        n_low_confidence=0,
        n_failed=0,
    )
    revised_run = run.model_copy(update={
        'config': run.config.model_copy(update={'labels': [revised_spec]}),
        'traces': [revised_trace],
        'labels': {'failure_taxonomy': revised},
    })
    revised_path = save_run(revised_run, tmp_path)
    revised_restored = load_run(revised_path)
    assert revised_restored.traces[0].labels['failure_taxonomy'].value == 'No failure'
    assert revised_restored.labels['failure_taxonomy'].counts_warning is None


def test_map_layouts_survive_save_and_legacy_reports_remain_readable(minimal_run, tmp_path: Path) -> None:
    import json

    from evaluatorq.insights.models import TraceInsight

    minimal_run.traces = [TraceInsight(
        trace_id='map-trace',
        coords={'intent': (1.0, 2.0, 3.0)},
        coords_2d={'intent': (9.0, 8.0)},
    )]
    path = save_run(run=minimal_run, runs_dir=tmp_path)
    restored = load_run(path=path)

    assert restored.traces[0].coords == {'intent': (1.0, 2.0, 3.0)}
    assert restored.traces[0].coords_2d == {'intent': (9.0, 8.0)}
    raw = json.loads(path.read_text())
    raw['traces'][0].pop('coords_2d')
    path.write_text(json.dumps(raw))

    assert load_run(path=path).traces[0].coords_2d is None


def test_same_name_and_second_never_overwrites(tmp_path: Path, minimal_run) -> None:
    first = save_run(minimal_run, tmp_path)
    second_run = minimal_run.model_copy(update={'run_id': 'run-2'})
    second = save_run(second_run, tmp_path)

    assert first != second
    assert first.name == 'insights_20260901-000000_minimal-run.json'
    assert second.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(first).run_id == 'run-1'
    assert load_run(second).run_id == 'run-2'


@pytest.mark.skipif(os.name == 'nt', reason='Windows uses exclusive rename instead of the POSIX symlink fallback')
def test_save_run_falls_back_when_hard_links_are_unsupported(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)

    def unexpected_replace(self: Path, destination: Path) -> Path:
        raise AssertionError('fallback must never replace a claimed filename')

    monkeypatch.setattr(Path, 'replace', unexpected_replace)

    path = save_run(minimal_run, tmp_path)

    assert load_run(path) == minimal_run
    assert path.is_symlink()
    assert path.read_text(encoding='utf-8') == minimal_run.model_dump_json(indent=2) + '\n'


def test_fallback_symlink_is_readable_with_relative_runs_dir(tmp_path: Path, minimal_run, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)

    def unexpected_replace(self: Path, destination: Path) -> Path:
        raise AssertionError('fallback must never replace a claimed filename')

    monkeypatch.setattr(Path, 'replace', unexpected_replace)

    path = save_run(minimal_run, Path('runs'))
    assert load_run(path) == minimal_run


@pytest.mark.skipif(os.name == 'nt', reason='Windows uses exclusive rename instead of the POSIX symlink fallback')
def test_save_run_fallback_does_not_overwrite_name_claimed_during_publication(
    tmp_path: Path, minimal_run, monkeypatch
) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    original_symlink = Path.symlink_to
    raced_path = tmp_path / 'insights_20260901-000000_minimal-run.json'
    claimed = False

    def claim_before_symlink(path: Path, target: str, target_is_directory: bool = False) -> None:
        nonlocal claimed
        if Path(path) == raced_path and not claimed:
            claimed = True
            raced_path.write_text('claimed by another process', encoding='utf-8')
        original_symlink(path, target, target_is_directory)

    monkeypatch.setattr(Path, 'symlink_to', claim_before_symlink)

    saved = save_run(minimal_run, tmp_path)

    assert claimed
    assert raced_path.read_text(encoding='utf-8') == 'claimed by another process'
    assert saved.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(saved) == minimal_run


@pytest.mark.skipif(os.name == 'nt', reason='Windows uses exclusive rename instead of the POSIX symlink fallback')
def test_fallback_never_exposes_json_until_atomic_publication(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    original_symlink = Path.symlink_to
    observations: list[list[Path]] = []

    def inspect_before_publish(path: Path, target: str, target_is_directory: bool = False) -> None:
        observations.append(list_run_paths(tmp_path))
        original_symlink(path, target, target_is_directory)

    monkeypatch.setattr(Path, 'symlink_to', inspect_before_publish)
    path = save_run(minimal_run, tmp_path)

    assert observations == [[]]
    assert load_run(path) == minimal_run
    assert path.resolve().exists()


@pytest.mark.skipif(os.name == 'nt', reason='Windows uses exclusive rename instead of the POSIX symlink fallback')
def test_save_run_fallback_fails_cleanly_when_symlinks_are_unavailable(
    tmp_path: Path, minimal_run, monkeypatch
) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    def unsupported_symlink(target: str, target_is_directory: bool = False) -> None:
        raise OSError(errno.EOPNOTSUPP, 'symlinks are unsupported')

    monkeypatch.setattr(Path, 'symlink_to', unsupported_symlink)

    with pytest.raises(OSError, match='symlinks are unsupported'):
        save_run(minimal_run, tmp_path)

    assert list_run_paths(tmp_path) == []


def test_windows_exclusive_rename_fallback_handles_collision_without_symlinks(
    tmp_path: Path, minimal_run, monkeypatch
) -> None:
    def unsupported_link(*_args) -> None:
        raise OSError(errno.EOPNOTSUPP, 'no hard links')

    monkeypatch.setattr(os, 'link', unsupported_link)
    monkeypatch.setattr(store, '_rename_is_exclusive', lambda: True)
    monkeypatch.setattr(Path, 'symlink_to', lambda *_args: pytest.fail('Windows rename needs no symlink'))
    original_rename = Path.rename

    def exclusive_rename(source: Path, target: Path) -> Path:
        if target.exists():
            raise FileExistsError(target)
        return original_rename(source, target)

    monkeypatch.setattr(Path, 'rename', exclusive_rename)
    first = save_run(minimal_run, tmp_path)
    second = save_run(minimal_run.model_copy(update={'run_id': 'run-2'}), tmp_path)

    assert first.name == 'insights_20260901-000000_minimal-run.json'
    assert second.name == 'insights_20260901-000000_minimal-run-2.json'
    assert not first.is_symlink() and not second.is_symlink()
    assert load_run(first).run_id == 'run-1'
    assert load_run(second).run_id == 'run-2'


def test_save_run_fallback_does_not_overwrite_collision(tmp_path: Path, minimal_run, monkeypatch) -> None:
    original = tmp_path / 'insights_20260901-000000_minimal-run.json'
    original.write_text('keep this file', encoding='utf-8')

    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)

    saved = save_run(minimal_run, tmp_path)

    assert original.read_text(encoding='utf-8') == 'keep this file'
    assert saved.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(saved) == minimal_run


def test_save_run_does_not_hide_unrelated_link_errors(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def permission_denied(*args) -> None:
        raise OSError(errno.ENOSPC, 'disk is full')

    monkeypatch.setattr(os, 'link', permission_denied)

    with pytest.raises(OSError, match='disk is full'):
        save_run(minimal_run, tmp_path)
    assert list_run_paths(tmp_path) == []


def test_list_run_paths_is_newest_first_and_missing_directory_is_empty(tmp_path: Path) -> None:
    older = tmp_path / 'insights_older.json'
    newer = tmp_path / 'insights_newer.json'
    older.write_text('{}', encoding='utf-8')
    newer.write_text('{}', encoding='utf-8')
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert list_run_paths(tmp_path) == [newer, older]
    assert list_run_paths(tmp_path / 'missing') == []


def test_list_run_paths_keeps_listing_failure_non_fatal(tmp_path: Path, monkeypatch) -> None:
    def fail_glob(self, pattern):
        raise OSError('directory unavailable')

    monkeypatch.setattr(Path, 'glob', fail_glob)

    assert list_run_paths(tmp_path) == []
