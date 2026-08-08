"""Tests for the combined Dashboard landing, per-kind run lists, and the
metrics aggregation that feeds them (RES-974)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import metrics
from evaluatorq.dashboard.app import build_app


def _redteam_payload(
    name: str,
    *,
    created: str,
    resistance: float,
    vulns: int,
    evaluated: int,
    tokens: int,
    severity: dict[str, int],
) -> dict:
    return {
        'pipeline': {'mode': 'adaptive'},
        'created_at': created,
        'run_name': name,
        'total_results': evaluated,
        'results': [
            {
                'attack': {'severity': 'critical', 'strategy_name': 'direct_override'},
                'agent': {'display_name': 'Refund agent', 'model': 'gpt-5.4'},
                'vulnerable': True,
                'error': None,
            },
            {
                'attack': {'severity': 'low', 'strategy_name': 'roleplay'},
                'agent': {'display_name': 'Refund agent', 'model': 'gpt-5.4'},
                'vulnerable': False,
                'error': None,
            },
        ],
        'summary': {
            'resistance_rate': resistance,
            'vulnerabilities_found': vulns,
            'evaluated_attacks': evaluated,
            'token_usage_total': {'total_tokens': tokens, 'cost_usd': 0.0048},
            'by_severity': {k: {'vulnerabilities_found': v} for k, v in severity.items()},
        },
    }


def _legacy_redteam_payload(name: str, *, created: str, resistance: float, results: list[dict]) -> dict:
    """A red-team report predating summary.evaluated_attacks / vulnerabilities_found.

    Only resistance_rate is stored; the real counts, severities and token usage
    all live in the results list.
    """
    return {
        'pipeline': {'mode': 'adaptive'},
        'created_at': created,
        'run_name': name,
        'total_results': len(results),
        'results': results,
        'summary': {'resistance_rate': resistance},
    }


def _legacy_result(
    *,
    vulnerable: bool = False,
    error: str | None = None,
    severity: str = 'low',
    evaluation: dict | None = None,
    tokens: int = 0,
) -> dict:
    """One legacy result row.

    Omitting *evaluation* reproduces the oldest reports, which carry only the
    ``vulnerable`` flag; passing one exercises the authoritative
    ``evaluation.passed`` classification.
    """
    res: dict = {'attack': {'severity': severity}, 'vulnerable': vulnerable, 'error': error}
    if evaluation is not None:
        res['evaluation'] = evaluation
    if tokens:
        res['execution'] = {'token_usage': {'total_tokens': tokens, 'cost_usd': tokens * 0.001}}
    return res


def _sim_payload(name: str, *, created: str, averages: dict[str, float], n: int, tok_each: int) -> dict:
    return {
        'mode': 'run',
        'created_at': created,
        'run_name': name,
        'total_results': n,
        'scorer_averages': averages,
        'results': [{'total_tokens': tok_each} for _ in range(n)],
    }


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    (rt / 'refund_20260624_101500.json').write_text(
        json.dumps(
            _redteam_payload(
                'Refund agent probe',
                created='2026-06-24T10:15:00',
                resistance=0.86,
                vulns=18,
                evaluated=128,
                tokens=412000,
                severity={'critical': 3, 'high': 7, 'medium': 6, 'low': 2},
            )
        )
    )
    (sim / 'support_20260625_140000.json').write_text(
        json.dumps(
            _sim_payload(
                'Support agent simulation',
                created='2026-06-25T14:00:00',
                averages={'helpfulness': 0.91, 'safety': 0.97},
                n=40,
                tok_each=1550,
            )
        )
    )
    return [rt, sim]


@pytest.fixture
def client(roots: list[Path]) -> TestClient:
    return TestClient(build_app(roots=roots))


class TestMetrics:
    def test_run_rows_kinds_and_scores(self, roots: list[Path]) -> None:
        rows = metrics.run_rows(roots)
        by_surface = {r.surface: r for r in rows}
        assert set(by_surface) == {'redteam', 'sim'}
        # Red team score is the resistance rate.
        assert by_surface['redteam'].score == pytest.approx(0.86)
        assert by_surface['redteam'].status == 'finished'  # lifecycle, not quality
        # Sim score is the mean of scorer averages.
        assert by_surface['sim'].score == pytest.approx((0.91 + 0.97) / 2)

    def test_landing_aggregates(self, roots: list[Path]) -> None:
        data = metrics.landing(roots)
        assert data.total_runs == 2
        assert data.redteam_runs == 1
        assert data.sim_runs == 1
        # resistant = evaluated - vulns = 128 - 18; vulnerable = 18.
        assert data.resistant == 110
        assert data.vulnerable == 18
        # Severity rolls up in display order, non-zero only.
        assert data.severity == [('critical', 3), ('high', 7), ('medium', 6), ('low', 2)]
        # Token usage split by kind: redteam total + sim per-result sum.
        assert dict(data.tokens_by_kind)['Red team'] == 412000
        assert dict(data.tokens_by_kind)['Agent sim'] == 40 * 1550
        assert len(data.recent) == 2

    def test_landing_includes_legacy_redteam_counts(self, roots: list[Path], tmp_path: Path) -> None:
        # A legacy report (no evaluated_attacks in summary) must contribute its
        # real counts, derived from results, to the landing aggregate (RES-1202).
        rt = roots[0]
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Legacy probe',
                    created='2026-01-01T00:00:00',
                    resistance=2 / 3,
                    results=[
                        _legacy_result(vulnerable=False),
                        _legacy_result(vulnerable=False),
                        _legacy_result(vulnerable=True),
                        # Errored result was never evaluated; must not be counted.
                        _legacy_result(vulnerable=False, error='timeout'),
                    ],
                )
            )
        )
        data = metrics.landing(roots)
        # Modern report: resistant 110, vulnerable 18. Legacy adds 2 and 1.
        assert data.resistant == 112
        assert data.vulnerable == 19
        assert data.resistance_rate == pytest.approx(112 / 131)

    def test_landing_legacy_without_results_is_not_counted(self, tmp_path: Path) -> None:
        # A rate without an attack count has no weight in the attack-weighted
        # aggregate: it stays out, and the aggregate stays consistent (None
        # rather than a fabricated number).
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(_legacy_redteam_payload('Rate only', created='2026-01-01T00:00:00', resistance=0.5, results=[]))
        )
        data = metrics.landing([rt])
        assert data.redteam_runs == 1  # the run itself is still listed
        assert data.resistant == 0
        assert data.vulnerable == 0
        assert data.resistance_rate is None

    def test_landing_legacy_severity_and_tokens_are_derived(self, tmp_path: Path) -> None:
        # A legacy run must weigh in on the severity bars and the cost totals
        # too, not just the donut — otherwise it is counted as a run whose
        # attacks and spend are invisible everywhere else on the page.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Legacy probe',
                    created='2026-01-01T00:00:00',
                    resistance=0.5,
                    results=[
                        _legacy_result(vulnerable=True, severity='critical', tokens=100),
                        _legacy_result(vulnerable=True, severity='low', tokens=100),
                        _legacy_result(vulnerable=False, tokens=200),
                    ],
                )
            )
        )
        data = metrics.landing([rt])
        assert dict(data.severity) == {'critical': 1, 'low': 1}
        assert dict(data.tokens_by_kind)['Red team'] == 400
        assert data.total_tokens == 400
        assert dict(data.cost_by_kind)['Red team'] == pytest.approx(0.4)

    def test_landing_evaluation_failure_is_not_counted_as_resistant(self, tmp_path: Path) -> None:
        # error carries the *target* error only. A result whose generation
        # succeeded but whose evaluation never produced a boolean was not
        # evaluated, and must not pad the resistant bucket.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Judge crashed',
                    created='2026-01-01T00:00:00',
                    resistance=1.0,
                    results=[
                        _legacy_result(evaluation={'passed': True}),
                        _legacy_result(evaluation={'passed': False}, severity='high'),
                        # Judge never returned a verdict: evaluation present but
                        # passed is None, and no target-side error to go on.
                        _legacy_result(evaluation={'passed': None}),
                    ],
                )
            )
        )
        data = metrics.landing([rt])
        assert data.resistant == 1
        assert data.vulnerable == 1
        assert data.resistance_rate == pytest.approx(0.5)

    def test_landing_empty(self, tmp_path: Path) -> None:
        empty = [tmp_path / 'runs', tmp_path / 'sim-runs']
        for p in empty:
            p.mkdir()
        data = metrics.landing(empty)
        assert data.total_runs == 0
        assert data.resistance_rate is None


class TestLandingScreen:
    def test_dashboard_landing_renders(self, client: TestClient) -> None:
        r = client.get('/')
        assert r.status_code == 200
        assert 'stat-band' in r.text
        # Attack-resistance donut was dropped; band now leads with Jobs run.
        assert 'Attack resistance' not in r.text
        assert 'Jobs run' in r.text
        assert 'Findings by severity' in r.text
        # Recent runs include both run names.
        assert 'Refund agent probe' in r.text
        assert 'Support agent simulation' in r.text
        # The combined dashboard mixes surfaces; the Type column disambiguates
        # (surface glyph + label) instead of an inline bubble, rendered as an
        # airy column list rather than a boxed table.
        assert 'type-cell' in r.text
        assert 'recent-runs' in r.text

    def test_dashboard_nav_active(self, client: TestClient) -> None:
        r = client.get('/')
        assert '<a class="nav-item active" href="/"' in r.text

    def test_redteam_overview(self, client: TestClient) -> None:
        # Red Team is the design's rich overview: KPI band + run-level table,
        # one row per red team run (not per attack).
        r = client.get('/?surface=redteam')
        assert r.status_code == 200
        assert 'kpi-band' in r.text
        assert 'Attacks run' in r.text
        assert 'Recent runs' in r.text
        # Run-level rows surface the run name; the sim run must not leak.
        assert 'Refund agent' in r.text
        assert 'Support agent simulation' not in r.text
        assert '<span class="kind-badge' not in r.text

    def test_agentsim_overview(self, client: TestClient) -> None:
        # Agent Sim is the design's rich overview: KPI band + run-level table,
        # one row per simulation run (not per simulation case).
        r = client.get('/?surface=sim')
        assert r.status_code == 200
        assert 'kpi-band' in r.text
        assert 'Simulations run' in r.text
        assert 'Recent runs' in r.text
        # The red team run must not leak onto the sim surface.
        assert 'Refund agent probe' not in r.text
        assert '<span class="kind-badge' not in r.text

    def test_unknown_surface_empty(self, client: TestClient) -> None:
        r = client.get('/?surface=bogus')
        assert r.status_code == 200
        assert 'no reports' in r.text.lower()

    def test_settings_config(self, client: TestClient) -> None:
        r = client.get('/settings')
        assert r.status_code == 200
        # Read-only runtime config, not the stub.
        assert 'Configuration' in r.text
        assert 'Run stores' in r.text
        assert 'API key' in r.text
        assert '<a class="nav-item active" href="/settings"' in r.text

    def test_mask_key_shows_only_suffix(self) -> None:
        from evaluatorq.dashboard.app import _mask_key

        # Only the last 4 chars are revealed; the rest is starred out and capped.
        assert _mask_key('sk-proj-abcdEF1234wxyz') == '****************wxyz'
        assert 'abcdEF' not in _mask_key('sk-proj-abcdEF1234wxyz')
        # Too-short keys reveal nothing but length.
        assert _mask_key('abcd') == '****'

    def test_global_search(self, client: TestClient) -> None:
        # The topbar search box was removed, but the /search route still works.
        assert 'class="search-input"' not in client.get('/').text
        # The search fragment matches report names case-insensitively.
        r = client.get('/search', params={'q': 'refund'})
        assert r.status_code == 200
        assert 'Refund agent probe' in r.text
        assert 'search-hit' in r.text
        # Empty query returns nothing; no-match returns a friendly message.
        assert client.get('/search', params={'q': ''}).text.strip() == ''
        assert 'No matching reports' in client.get('/search', params={'q': 'zzzzz'}).text


class TestReportHeader:
    def test_report_view_has_back_link_and_export(self, tmp_path: Path) -> None:
        # Use a fully-valid RedTeamReport so the report view renders (the broken
        # branch returns early without the header chrome).
        from tests.dashboard.test_downloads import _make_rt_report

        from evaluatorq.dashboard.library import report_id

        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        rt_file = rt / 'rt_header_test.json'
        rt_file.write_text(_make_rt_report().model_dump_json())

        client = TestClient(build_app(roots=[rt, sim]))
        rid = report_id(rt_file)
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        assert 'class="report-back"' in r.text
        assert 'Red team runs' in r.text
        # Topbar Export action points at the standalone HTML export.
        assert f'/r/{rid}/export.html' in r.text
