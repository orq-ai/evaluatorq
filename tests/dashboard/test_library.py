from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluatorq.dashboard.library import ReportCard, report_id, resolve, scan, sniff_kind


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _redteam_payload() -> dict:
    return {
        'version': '2.0.0',
        'created_at': datetime.now(tz=timezone.utc).isoformat(),
        'pipeline': 'static',
        'categories_tested': ['ASI01'],
        'total_results': 0,
        'results': [],
        'summary': {},
    }


def _sim_payload() -> dict:
    return {
        'run_name': 'demo',
        'created_at': datetime.now(tz=timezone.utc).isoformat(),
        'mode': 'run',
        'target_kind': 'orq_agent',
        'evaluator_names': [],
        'total_results': 0,
        'scorer_averages': {},
        'results': [],
    }


def test_report_id_is_stable_and_urlsafe():
    p = Path('/tmp/some/run_20260101_000000.json')
    rid = report_id(p)
    assert rid == report_id(p)
    assert '/' not in rid and ' ' not in rid


def test_sniff_kind_discriminates_both_surfaces():
    assert sniff_kind(_redteam_payload()) == 'redteam'
    assert sniff_kind(_sim_payload()) == 'sim'
    assert sniff_kind({'unrelated': True}) is None


def test_sniff_kind_overlapping_payloads_resolve_by_priority():
    # A sim run carrying a stray 'pipeline' must still sniff sim (mode checked first).
    sim_with_pipeline = {**_sim_payload(), 'pipeline': 'static'}
    assert sniff_kind(sim_with_pipeline) == 'sim'
    # A redteam report has no 'mode', so it never crosses to sim.
    assert sniff_kind(_redteam_payload()) == 'redteam'


def test_scan_lists_both_surfaces_and_excludes_artifacts(tmp_path):
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    _write(rt / 'redteam_20260101_000000.json', _redteam_payload())
    _write(sim / 'sim_20260101_000000.json', _sim_payload())
    _write(rt / '01_objectives.json', {'saved_at': 'x', 'data': {}})
    cards = scan([rt, sim])
    assert sorted(c.surface for c in cards) == ['redteam', 'sim']
    assert all(isinstance(c, ReportCard) for c in cards)


def test_scan_dedupes_overlapping_roots(tmp_path):
    """The same directory reached through two roots (repeated arg, relative +
    absolute forms, or one arg inside another's expansion) must not
    double-count its reports — the landing rollups (jobs, spend, costed runs)
    read straight off this list."""
    rt = tmp_path / 'runs'
    rt.mkdir()
    _write(rt / 'redteam_20260101_000000.json', _redteam_payload())

    same_root_twice = scan([rt, rt])
    assert len(same_root_twice) == 1

    resolved_and_symlinkless_alias = scan([rt, tmp_path / '.' / 'runs'])
    assert len(resolved_and_symlinkless_alias) == 1


def test_resolve_roundtrips_and_misses_to_none(tmp_path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    p = rt / 'redteam_20260101_000000.json'
    _write(p, _redteam_payload())
    assert resolve(report_id(p), [rt]) == p
    assert resolve('deadbeef', [rt]) is None


def test_load_model_cached_reuses_object_until_mtime_changes(tmp_path):
    """A validated model is returned from cache (same object) on repeat loads,
    and a file rewrite (new mtime) invalidates it — no stale data served."""
    import os

    from evaluatorq.dashboard.library import load_model_cached

    calls = {'n': 0}

    class _Model:
        @classmethod
        def model_validate(cls, data):
            calls['n'] += 1
            obj = _Model()
            obj.data = data
            return obj

    p = tmp_path / 'm.json'
    p.write_text('{"a": 1}')
    first = load_model_cached(p, _Model.model_validate)
    second = load_model_cached(p, _Model.model_validate)
    assert first is second  # cache hit — no re-validation
    assert calls['n'] == 1

    # Rewrite with a bumped mtime → cache miss → fresh validation.
    p.write_text('{"a": 2}')
    os.utime(p, ns=(0, p.stat().st_mtime_ns + 1_000_000))
    third = load_model_cached(p, _Model.model_validate)
    assert third is not first
    assert third.data == {'a': 2}
    assert calls['n'] == 2


def test_broken_report_surfaces_as_card_not_skipped(tmp_path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    # sniffs redteam (pipeline present) but missing 'summary' -> broken card.
    _write(rt / 'broken_20260101_000000.json', {'pipeline': 'static', 'results': []})
    (rt / 'garbage_20260101_000000.json').write_text('{not json')  # unparseable -> skipped
    cards = scan([rt])
    assert len(cards) == 1
    assert cards[0].error is not None


def _start_manifest(runs_dir: Path, run_id: str, surface: str, name: str):
    from evaluatorq.common.run_manifest import start_manifest

    return start_manifest(run_id=run_id, surface=surface, run_name=name, runs_dir=runs_dir)


def _rt_summary(total: int) -> dict:
    """The summary shape a real red-team run writes (RedTeamReport.manifest_summary).

    A summary carrying only ``total_results`` reads as incomplete and is demoted
    to a full-report read, so manifest-first tests must use the full shape.
    """
    return {
        'pipeline': 'static',
        'total_results': total,
        'total_attacks': total,
        'vulnerability_rate': 0.0,
        'resistance_rate': 1.0,
        'tested_agents': [],
    }


def test_scan_builds_card_from_manifest_without_reading_report(tmp_path):
    # report_path points at a NON-EXISTENT file; the card must still build from
    # the manifest's compact summary (proving no full-report read happens).
    rt = tmp_path / 'runs'
    rt.mkdir()
    bogus = rt / 'nonexistent_20260101.json'  # deliberately not written
    w = _start_manifest(rt, 'm1', 'redteam', 'from-manifest')
    w.complete(report_path=bogus, summary=_rt_summary(7))

    cards = scan([rt])
    assert len(cards) == 1
    card = cards[0]
    assert card.name == 'from-manifest'
    assert card.surface == 'redteam'
    assert card.status == 'completed'
    assert card.headline == '7 attacks'
    assert not bogus.exists()


def test_scan_legacy_reports_only_lists_as_before(tmp_path):
    # A runs dir with ONLY reports (no .manifests) must list exactly as today.
    rt = tmp_path / 'runs'
    rt.mkdir()
    _write(rt / 'redteam_20260101_000000.json', _redteam_payload())
    cards = scan([rt])
    assert len(cards) == 1
    assert cards[0].surface == 'redteam'
    assert cards[0].status is None  # legacy cards carry no manifest status
    assert cards[0].path is not None


def test_scan_backfills_manifest_for_legacy_report(tmp_path):
    # Second scan must serve the same card from the sidecar, without the report.
    rt = tmp_path / 'runs'
    rt.mkdir()
    report = rt / 'redteam_20260101_000000.json'
    payload = _redteam_payload()
    payload['total_results'] = 4
    _write(report, payload)

    first = scan([rt])
    assert (rt / '.manifests' / 'redteam_20260101_000000.json').exists()

    report.unlink()  # proves the second scan reads only the manifest
    second = scan([rt])
    assert len(second) == 1
    assert second[0].id == first[0].id
    assert second[0].name == first[0].name
    assert second[0].headline == first[0].headline == '4 attacks'
    assert second[0].status == 'completed'


def test_scan_backfill_skips_broken_report(tmp_path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    payload = _redteam_payload()
    del payload['summary']  # missing required field -> card carries an error
    _write(rt / 'redteam_20260101_000000.json', payload)

    cards = scan([rt])
    assert cards[0].error
    assert not (rt / '.manifests').exists()


def test_scan_mixed_dir_dedups_by_report_path(tmp_path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    report = rt / 'covered_20260101_000000.json'
    _write(report, _redteam_payload())
    w = _start_manifest(rt, 'cov', 'redteam', 'covered')
    w.complete(report_path=report, summary=_rt_summary(3))
    # A legacy report with no manifest alongside it.
    _write(rt / 'legacy_20250101_000000.json', _redteam_payload())

    cards = scan([rt])
    # The covered report is not listed twice; total = manifest card + legacy card.
    assert len(cards) == 2
    names = sorted(c.name for c in cards)
    assert names == ['covered', 'legacy_20250101_000000']
    covered = next(c for c in cards if c.name == 'covered')
    assert covered.status == 'completed'


def test_scan_in_flight_manifest_appears_with_status_and_stage(tmp_path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    w = _start_manifest(rt, 'live', 'sim', 'in-flight')
    w.start_stage('Simulating')

    cards = scan([rt])
    assert len(cards) == 1
    card = cards[0]
    assert card.path is None
    assert card.status == 'running'
    assert card.stage == 'Simulating'
    # An in-flight card resolves to its manifest, not a report.
    from evaluatorq.dashboard.library import resolve, resolve_manifest

    assert resolve(card.id, [rt]) is None
    manifest = resolve_manifest(card.id, [rt])
    assert manifest is not None
    assert manifest.run_name == 'in-flight'


def test_scan_errored_manifest_without_report_is_listed(tmp_path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    w = _start_manifest(rt, 'boom', 'redteam', 'errored')
    w.fail('kaboom')

    cards = scan([rt])
    assert len(cards) == 1
    assert cards[0].status == 'error'
    assert cards[0].path is None


def test_fingerprint_changes_when_an_in_flight_manifest_advances(tmp_path):
    """Reports are write-once, but an in-flight run rewrites its manifest in
    place — the fingerprint must see that or a running run's row would freeze."""
    from evaluatorq.dashboard.library import fingerprint

    rt = tmp_path / 'runs'
    rt.mkdir()
    w = _start_manifest(rt, 'live', 'sim', 'in-flight')
    before = fingerprint([rt])
    w.start_stage('generating')
    assert fingerprint([rt]) != before


def test_fingerprint_changes_when_a_report_lands(tmp_path):
    from evaluatorq.dashboard.library import fingerprint

    rt = tmp_path / 'runs'
    rt.mkdir()
    before = fingerprint([rt])
    _write(rt / 'redteam_20260101_000000.json', _redteam_payload())
    assert fingerprint([rt]) != before


def test_fingerprint_ignores_stage_artifacts(tmp_path):
    """Stage artifacts (``01_``…) are excluded from every aggregate, so rewriting
    one must not invalidate the caches keyed on this fingerprint."""
    from evaluatorq.dashboard.library import fingerprint

    rt = tmp_path / 'runs'
    rt.mkdir()
    _write(rt / 'redteam_20260101_000000.json', _redteam_payload())
    before = fingerprint([rt])
    _write(rt / '01_objectives.json', {'objectives': []})
    assert fingerprint([rt]) == before


def test_backfilled_manifest_carries_the_full_writer_summary(tmp_path):
    """The sidecar must hold every field `eq redteam runs` reads, not just the
    one the dashboard card renders."""
    from evaluatorq.redteam.contracts import RedTeamReport

    rt = tmp_path / 'runs'
    rt.mkdir()
    payload = _redteam_payload()
    payload['total_results'] = 4
    payload['tested_agents'] = ['agent:x']
    payload['summary'] = {'total_attacks': 4, 'vulnerability_rate': 0.25, 'resistance_rate': 0.75}
    _write(rt / 'redteam_20260101_000000.json', payload)

    scan([rt])

    written = json.loads((rt / '.manifests' / 'redteam_20260101_000000.json').read_text())
    assert written['summary'] == RedTeamReport.model_validate(payload).manifest_summary()


def test_scan_rewrites_a_thin_sidecar(tmp_path):
    """A sidecar written by the first backfill version holds only total_results;
    it must be re-read and upgraded rather than served as-is forever."""
    rt = tmp_path / 'runs'
    rt.mkdir()
    payload = _redteam_payload()
    payload['total_results'] = 4
    payload['summary'] = {'total_attacks': 4, 'vulnerability_rate': 0.25, 'resistance_rate': 0.75}
    report = rt / 'redteam_20260101_000000.json'
    _write(report, payload)
    _start_manifest(rt, report.stem, 'redteam', 'thin').complete(
        report_path=report, summary={'total_results': 4}
    )

    cards = scan([rt])

    assert len(cards) == 1  # not listed twice
    written = json.loads((rt / '.manifests' / f'{report.stem}.json').read_text())
    assert written['summary']['total_attacks'] == 4
    assert written['summary']['pipeline'] == 'static'
