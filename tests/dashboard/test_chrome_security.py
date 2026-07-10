"""Chrome-layer regression tests: index-card name / page-title XSS and the
naive-vs-aware created_at index-sort crash. These sites live outside the
per-surface view builders and were missed by per-task review."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id


def _redteam(created_at: str, description: str = 'demo') -> dict:
    return {
        'version': '2.0.0',
        'created_at': created_at,
        'description': description,
        'pipeline': 'static',
        'categories_tested': [],
        'total_results': 0,
        'results': [],
        'summary': {},
    }


def _client(roots: list[Path]) -> TestClient:
    return TestClient(build_app(roots))


def test_index_escapes_report_name(tmp_path: Path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    payload = _redteam(datetime.now(tz=timezone.utc).isoformat(), description="<script>alert(1)</script>")
    (rt / 'rt_20260101_000000.json').write_text(json.dumps(payload))
    r = _client([rt]).get('/')
    assert r.status_code == 200
    assert '<script>alert(1)</script>' not in r.text
    assert '&lt;script&gt;' in r.text


def test_report_title_escapes_description(tmp_path: Path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    payload = _redteam(
        datetime.now(tz=timezone.utc).isoformat(),
        description="</title><script>alert(1)</script>",
    )
    p = rt / 'rt_20260101_000000.json'
    p.write_text(json.dumps(payload))
    r = _client([rt]).get(f'/r/{report_id(p)}')
    assert r.status_code == 200
    assert '</title><script>alert(1)</script>' not in r.text


def test_index_does_not_crash_on_mixed_naive_and_aware_timestamps(tmp_path: Path):
    rt = tmp_path / 'runs'
    rt.mkdir()
    # naive created_at (no tz offset) alongside a report that falls back to aware.
    (rt / 'naive_20260101_000000.json').write_text(json.dumps(_redteam('2026-01-01T00:00:00')))
    (rt / 'missing_20260102_000000.json').write_text(
        json.dumps({k: v for k, v in _redteam('').items() if k != 'created_at'})
    )
    r = TestClient(build_app([rt]), raise_server_exceptions=True).get('/')
    assert r.status_code == 200


def _sim_run_with_xss_criterion():
    from evaluatorq.contracts import Message, TokenUsage
    from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy

    result = SimulationResult(
        messages=[Message(role='user', content='hi')],
        terminated_by=TerminatedBy.judge,
        reason='<img src=x onerror=alert(1)>',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        turn_metrics=[],
        token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        metadata={'persona': 'alice', 'scenario': 'demo'},
    )
    return SimulationRun(
        run_name='xss-criteria-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=1,
        scorer_averages={'goal_achieved': 1.0},
        results=[result],
    )


def test_transcript_criteria_escapes_html(tmp_path: Path):
    """Task 12: transcript fragment criteria/judge markup must escape HTML —
    both the judge_reason and any criteria description are report-derived
    strings and stored-XSS vectors."""
    sim_dir = tmp_path / 'sim-runs'
    sim_dir.mkdir()
    run = _sim_run_with_xss_criterion()
    p = sim_dir / 'xss_criteria.json'
    p.write_text(run.model_dump_json())

    client = TestClient(build_app([sim_dir]), raise_server_exceptions=True)
    rid = report_id(p)
    r = client.get(f'/r/{rid}/sim/transcript?idx=0')
    assert r.status_code == 200
    assert '<img src=x onerror=alert(1)>' not in r.text
    assert '&lt;img src=x onerror=alert(1)&gt;' in r.text
