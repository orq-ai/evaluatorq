from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluatorq.dashboard.library import ReportCard, report_id, resolve, scan, sniff_kind


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _redteam_payload() -> dict:
    return {
        'version': '2.0.0', 'created_at': datetime.now(tz=timezone.utc).isoformat(),
        'pipeline': 'static', 'categories_tested': ['ASI01'], 'total_results': 0,
        'results': [], 'summary': {},
    }


def _sim_payload() -> dict:
    return {
        'run_name': 'demo', 'created_at': datetime.now(tz=timezone.utc).isoformat(),
        'mode': 'run', 'target_kind': 'orq_agent', 'evaluator_names': [],
        'total_results': 0, 'scorer_averages': {}, 'results': [],
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
    rt = tmp_path / 'runs'; sim = tmp_path / 'sim-runs'
    rt.mkdir(); sim.mkdir()
    _write(rt / 'redteam_20260101_000000.json', _redteam_payload())
    _write(sim / 'sim_20260101_000000.json', _sim_payload())
    _write(rt / '01_objectives.json', {'saved_at': 'x', 'data': {}})
    cards = scan([rt, sim])
    assert sorted(c.surface for c in cards) == ['redteam', 'sim']
    assert all(isinstance(c, ReportCard) for c in cards)


def test_resolve_roundtrips_and_misses_to_none(tmp_path):
    rt = tmp_path / 'runs'; rt.mkdir()
    p = rt / 'redteam_20260101_000000.json'
    _write(p, _redteam_payload())
    assert resolve(report_id(p), [rt]) == p
    assert resolve('deadbeef', [rt]) is None


def test_broken_report_surfaces_as_card_not_skipped(tmp_path):
    rt = tmp_path / 'runs'; rt.mkdir()
    # sniffs redteam (pipeline present) but missing 'summary' -> broken card.
    _write(rt / 'broken_20260101_000000.json', {'pipeline': 'static', 'results': []})
    (rt / 'garbage_20260101_000000.json').write_text('{not json')  # unparseable -> skipped
    cards = scan([rt])
    assert len(cards) == 1
    assert cards[0].error is not None
