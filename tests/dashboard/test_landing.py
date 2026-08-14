"""Tests for the combined Dashboard landing, per-kind run lists, and the
metrics aggregation that feeds them (RES-974)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import metrics, view
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

    def test_legacy_row_keeps_its_recorded_rate_when_nothing_is_derivable(self, tmp_path: Path) -> None:
        # Deriving nothing is not the same as measuring zero: a run with a
        # recorded rate but no per-attack results to re-derive from must keep
        # showing that rate rather than blanking its score.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(_legacy_redteam_payload('Rate only', created='2026-01-01T00:00:00', resistance=0.5, results=[]))
        )
        row = metrics.run_rows([rt])[0]
        assert row.score == pytest.approx(0.5)
        assert row.stored_score is None  # not re-derived, so nothing to reconcile

    def test_legacy_row_marks_a_rate_it_recalculated(self, tmp_path: Path) -> None:
        # The recorded rate was computed over every attack (1/2); the dashboard
        # counts evaluated-only (1/1). Both numbers stay visible.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Judge crashed',
                    created='2026-01-01T00:00:00',
                    resistance=0.5,
                    results=[
                        _legacy_result(evaluation={'passed': True}),
                        _legacy_result(evaluation={'passed': None}),
                    ],
                )
            )
        )
        row = metrics.run_rows([rt])[0]
        assert row.score == pytest.approx(1.0)
        assert row.stored_score == pytest.approx(0.5)
        assert (row.evaluated, row.attacks) == (1, 2)

    def test_legacy_row_is_unmarked_when_the_derivation_agrees(self, tmp_path: Path) -> None:
        # An indicator on every legacy run would be noise. Only a rate that
        # actually moved is worth flagging.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Agrees',
                    created='2026-01-01T00:00:00',
                    resistance=0.5,
                    results=[
                        _legacy_result(evaluation={'passed': True}),
                        _legacy_result(evaluation={'passed': False}),
                    ],
                )
            )
        )
        row = metrics.run_rows([rt])[0]
        assert row.score == pytest.approx(0.5)
        assert row.stored_score is None

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

    def test_landing_legacy_row_score_matches_derived_donut(self, tmp_path: Path) -> None:
        # A legacy stored resistance_rate may use a different denominator than
        # the evaluated-only one the donut derives. The run's row must go
        # through the same classifier, so one run cannot show two different
        # numbers on the same screen.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Total-denominator rate',
                    created='2026-01-01T00:00:00',
                    # Stored rate computed over all 4 results (2/4); the honest
                    # evaluated-only rate is 2/3.
                    resistance=0.5,
                    results=[
                        _legacy_result(vulnerable=False),
                        _legacy_result(vulnerable=False),
                        _legacy_result(vulnerable=True),
                        _legacy_result(vulnerable=False, error='timeout'),
                    ],
                )
            )
        )
        data = metrics.landing([rt])
        row = next(r for r in data.recent if r.surface == 'redteam')
        assert row.score == pytest.approx(2 / 3)
        assert data.resistance_rate == pytest.approx(2 / 3)

    def test_landing_legacy_result_without_vulnerable_key_is_unknown(self, tmp_path: Path) -> None:
        # Oldest schema, truncated record: no evaluation, no error, and no
        # vulnerable key either. "We don't know" must not land in the resistant
        # bucket — that is the optimistic-bias failure mode, one schema
        # generation older than the evaluation.passed one.
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Truncated record',
                    created='2026-01-01T00:00:00',
                    resistance=1.0,
                    results=[
                        {'attack': {'severity': 'low'}},
                        _legacy_result(vulnerable=True, severity='high'),
                    ],
                )
            )
        )
        data = metrics.landing([rt])
        assert data.resistant == 0
        assert data.vulnerable == 1
        assert data.resistance_rate == pytest.approx(0.0)

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


class TestAvgCost:
    def test_costed_runs_counted(self, roots: list[Path]) -> None:
        data = metrics.landing(roots)
        # Only the redteam fixture records cost_usd; the sim results carry tokens only.
        assert data.costed_runs == 1
        assert data.total_cost == pytest.approx(0.0048)

    def test_avg_cost_averages_over_costed_runs_only(self) -> None:
        data = metrics.Landing(
            total_runs=3,
            redteam_runs=2,
            sim_runs=1,
            resistance_rate=None,
            total_tokens=0,
            by_kind=[('Red team', 2), ('Agent sim', 1)],
            severity=[],
            tokens_by_kind=[],
            resistant=0,
            vulnerable=0,
            total_cost=0.30,
            costed_runs=1,
            cost_by_kind=[('Agent sim', 0.30)],
        )
        html = view.landing_body(data)
        # $0.30 / 1 costed run, NOT $0.30 / 3 total runs ($0.1000).
        assert '$0.3000' in html
        assert '$0.1000' not in html

    def test_avg_cost_na_when_nothing_records_cost(self) -> None:
        data = metrics.Landing(
            total_runs=2,
            redteam_runs=2,
            sim_runs=0,
            resistance_rate=None,
            total_tokens=0,
            by_kind=[('Red team', 2)],
            severity=[],
            tokens_by_kind=[],
            resistant=0,
            vulnerable=0,
            total_cost=None,
            costed_runs=0,
            cost_by_kind=[],
        )
        html = view.landing_body(data)
        # Unknown cost renders as the "not reported" em dash, never $0.00 —
        # a zero-cost run would be indistinguishable from an unrecorded one.
        assert '—' in html
        assert '$0.00' not in html


class TestUnknownCostNeverRendersAsZero:
    """Regression coverage for the None-vs-0.0 conflation bug found in Task 3's
    review: ``contracts.Usage``'s ``@model_serializer`` always emits the
    ``cost_usd`` key (even when the underlying ``total_cost`` is ``None``), so
    a bare ``'cost_usd' in usage`` membership check can no longer distinguish
    "cost recorded" from "cost unknown". An unknown cost must not be counted
    as costed, must not contribute to any spend total, and must render as the
    em dash — never ``$0.00``."""

    def _redteam_payload_null_cost(self, name: str, *, created: str) -> dict:
        return {
            'pipeline': {'mode': 'adaptive'},
            'created_at': created,
            'run_name': name,
            'total_results': 1,
            'results': [
                {
                    'attack': {'severity': 'low', 'strategy_name': 'roleplay'},
                    'agent': {'display_name': 'Refund agent', 'model': 'gpt-5.4'},
                    'vulnerable': False,
                    'error': None,
                }
            ],
            'summary': {
                'resistance_rate': 1.0,
                'vulnerabilities_found': 0,
                'evaluated_attacks': 1,
                # The serializer always emits 'cost_usd', but a provider that
                # never reported cost leaves it null — not absent.
                'token_usage_total': {'total_tokens': 500, 'cost_usd': None},
                'by_severity': {},
            },
        }

    def test_landing_does_not_count_null_cost_as_costed(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(json.dumps(self._redteam_payload_null_cost('P', created='2026-06-29T10:00:00')))

        data = metrics.landing([rt, sim])
        assert data.costed_runs == 0
        assert data.total_cost is None

        html = view.landing_body(data)
        assert '$0.00' not in html
        assert '—' in html

    def test_redteam_overview_does_not_count_null_cost(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(json.dumps(self._redteam_payload_null_cost('P', created='2026-06-29T10:00:00')))

        ov = metrics.redteam_overview([rt, sim])
        assert ov.total_cost is None
        assert ov.recent[0].cost is None

        html = view.redteam_overview_body(ov)
        assert '$0.00' not in html

    def test_sim_overview_does_not_count_null_cost(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (sim / 's.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-30T10:00:00',
                'run_name': 'S',
                'total_results': 1,
                'scorer_averages': {},
                'results': [
                    # Serializer-emitted null cost, not a missing key.
                    {'token_usage': {'total_tokens': 100, 'cost_usd': None}, 'goal_achieved': True, 'turn_count': 1}
                ],
            })
        )

        ov = metrics.sim_overview([rt, sim])
        assert ov.avg_cost is None
        assert ov.recent[0].cost is None

        html = view.sim_overview_body(ov)
        assert '$0.00' not in html


class TestPreBreakdownReportDegradesGracefully:
    """Reports saved before the cost-breakdown fields existed carry only the
    legacy ``cost_usd`` key (no ``input_cost`` / ``output_cost``), or predate
    cost entirely. Both shapes must render — the known total (if any) shown
    plainly, with no fabricated breakdown and no crash."""

    def test_legacy_cost_usd_only_still_totals_and_shows_no_breakdown(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(
            json.dumps({
                'pipeline': {'mode': 'adaptive'},
                'created_at': '2026-06-29T10:00:00',
                'run_name': 'Legacy report',
                'total_results': 1,
                'results': [
                    {
                        'attack': {'severity': 'low', 'strategy_name': 'roleplay'},
                        'agent': {'display_name': 'Refund agent', 'model': 'gpt-5.4'},
                        'vulnerable': False,
                        'error': None,
                    }
                ],
                'summary': {
                    'resistance_rate': 1.0,
                    'vulnerabilities_found': 0,
                    'evaluated_attacks': 1,
                    # Pre-breakdown shape: only the legacy aggregate key.
                    'token_usage_total': {'total_tokens': 500, 'cost_usd': 0.0025},
                    'by_severity': {},
                },
            })
        )

        data = metrics.landing([rt, sim])
        assert data.total_cost == pytest.approx(0.0025)
        assert data.costed_runs == 1
        # No breakdown was ever recorded, so it must not be fabricated.
        assert data.total_input_cost is None
        assert data.total_output_cost is None

        html = view.landing_body(data)
        assert '$0.0025' in html
        assert 'in —' not in html  # no dangling breakdown sub-line at all

        ov = metrics.redteam_overview([rt, sim])
        assert ov.total_cost == pytest.approx(0.0025)
        assert ov.total_input_cost is None
        assert ov.total_output_cost is None
        rt_html = view.redteam_overview_body(ov)
        assert '(in' not in rt_html

    def test_pre_cost_report_has_no_cost_key_at_all(self, tmp_path: Path) -> None:
        """Reports predating cost tracking entirely — no ``cost_usd`` key."""
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(
            json.dumps({
                'pipeline': {'mode': 'adaptive'},
                'created_at': '2026-06-29T10:00:00',
                'run_name': 'Ancient report',
                'total_results': 1,
                'results': [
                    {
                        'attack': {'severity': 'low', 'strategy_name': 'roleplay'},
                        'agent': {'display_name': 'Refund agent', 'model': 'gpt-5.4'},
                        'vulnerable': False,
                        'error': None,
                    }
                ],
                'summary': {
                    'resistance_rate': 1.0,
                    'vulnerabilities_found': 0,
                    'evaluated_attacks': 1,
                    'token_usage_total': {'total_tokens': 500},
                    'by_severity': {},
                },
            })
        )

        data = metrics.landing([rt, sim])
        assert data.costed_runs == 0
        assert data.total_cost is None
        html = view.landing_body(data)
        assert '$0.00' not in html
        assert '—' in html


class TestBarsRounding:
    def test_tiny_share_shows_less_than_one_percent(self) -> None:
        html = view._bars([('Red team', 328), ('Agent sim', 1)], ['c1', 'c2'])
        assert '&lt;1%' in html
        assert '· 0%' not in html
        # the tiny row still gets a visible sliver of bar
        assert 'width:1%' in html

    def test_dominant_share_shows_more_than_99_percent(self) -> None:
        html = view._bars([('Red team', 328), ('Agent sim', 1)], ['c1', 'c2'])
        assert '&gt;99%' in html
        assert '· 100%' not in html

    def test_exact_shares_unchanged(self) -> None:
        html = view._bars([('A', 3), ('B', 1)], ['c1', 'c2'])
        assert '· 75%' in html
        assert '· 25%' in html

    def test_zero_value_row_stays_zero(self) -> None:
        html = view._bars([('A', 5), ('B', 0)], ['c1', 'c2'])
        assert '· 0%' in html
        assert 'width:0%' in html

    def test_dominant_partial_bar_is_not_full_width(self) -> None:
        """The width uses the unrounded share: 328/329 must not draw a
        full-width bar next to its '>99%' label."""
        html = view._bars([('Red team', 328), ('Agent sim', 1)], ['c1', 'c2'])
        assert 'width:100%' not in html
        assert 'width:99.7%' in html

    def test_full_share_still_draws_full_width(self) -> None:
        html = view._bars([('A', 5), ('B', 0)], ['c1', 'c2'])
        assert 'width:100%' in html


class TestZeroAttackScore:
    def test_zero_evaluated_attacks_has_no_score(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'empty_20260731_130000.json').write_text(
            json.dumps(
                _redteam_payload(
                    'empty run',
                    created='2026-07-31T13:00:00Z',
                    resistance=1.0,
                    vulns=0,
                    evaluated=0,
                    tokens=0,
                    severity={},
                )
            )
        )
        rows = metrics.run_rows([rt])
        assert len(rows) == 1
        # 0 attacks evaluated: 1.00 would read as a perfect score.
        assert rows[0].score is None

    def test_null_recorded_rate_has_no_score(self, tmp_path: Path) -> None:
        """An explicitly null rate stays null, even for a run with zero attacks.

        ``zero_evaluated_attacks`` deliberately says False for a run that attempted
        nothing (empty, not unscored), so the null has to survive on its own — the
        legacy 1.0 default applies only when the field is *absent*.
        """
        rt = tmp_path / 'runs'
        rt.mkdir()
        payload = _redteam_payload(
            'no attacks',
            created='2026-07-31T13:00:00Z',
            resistance=1.0,
            vulns=0,
            evaluated=0,
            tokens=0,
            severity={},
        )
        payload['summary']['resistance_rate'] = None
        payload['summary']['total_attacks'] = 0
        (rt / 'noattacks_20260731_130000.json').write_text(json.dumps(payload))

        rows = metrics.run_rows([rt])
        assert len(rows) == 1
        assert rows[0].score is None


class TestDashboardCostCoverage:
    """Spend figures the dashboard shows must carry the same lower-bound label
    the markdown/HTML reports render. A total summed over calls where only some
    reported a cost is a lower bound; showing it bare reads as authoritative."""

    def _redteam_payload(self, name: str, *, created: str, priced: int, calls: int, estimated: int = 0) -> dict:
        return {
            'pipeline': {'mode': 'adaptive'},
            'created_at': created,
            'run_name': name,
            'total_results': 1,
            'results': [
                {
                    'attack': {'severity': 'low', 'strategy_name': 'roleplay'},
                    'agent': {'display_name': 'Refund agent', 'model': 'gpt-5.4'},
                    'vulnerable': False,
                    'error': None,
                }
            ],
            'summary': {
                'resistance_rate': 1.0,
                'vulnerabilities_found': 0,
                'evaluated_attacks': 1,
                'token_usage_total': {
                    'total_tokens': 500,
                    'cost_usd': 0.5,
                    'calls': calls,
                    'priced_calls': priced,
                    'estimated_calls': estimated,
                },
                'by_severity': {},
            },
        }

    def _roots(self, tmp_path: Path, payload: dict) -> tuple[Path, Path]:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(json.dumps(payload))
        return rt, sim

    def test_partial_coverage_labelled_on_landing_and_redteam(self, tmp_path: Path) -> None:
        rt, sim = self._roots(tmp_path, self._redteam_payload('P', created='2026-06-29T10:00:00', priced=3, calls=10))

        data = metrics.landing([rt, sim])
        assert (data.coverage.priced_calls, data.coverage.cost_calls) == (3, 10)
        assert '(3 of 10 calls)' in view.landing_body(data)

        ov = metrics.redteam_overview([rt, sim])
        assert (ov.coverage.priced_calls, ov.coverage.cost_calls) == (3, 10)
        assert '(3 of 10 calls)' in view.redteam_overview_body(ov)

    def test_all_estimated_store_labels_estimated(self, tmp_path: Path) -> None:
        """A store of catalogue-estimated runs (every priced call client-side)
        must aggregate to the "(estimated)" provenance label, not read as
        billed just because ``estimated_calls`` defaults to 0 on an unrelated
        code path."""
        rt, sim = self._roots(
            tmp_path,
            self._redteam_payload('P', created='2026-06-29T10:00:00', priced=10, calls=10, estimated=10),
        )

        data = metrics.landing([rt, sim])
        assert data.coverage.priced_calls == data.coverage.estimated_calls == 10
        assert '(estimated)' in view.landing_body(data)

        ov = metrics.redteam_overview([rt, sim])
        assert ov.coverage.priced_calls == ov.coverage.estimated_calls == 10
        assert '(estimated)' in view.redteam_overview_body(ov)

    def test_mixed_store_labels_partly_estimated(self, tmp_path: Path) -> None:
        """One fully-billed run beside one fully-estimated run must aggregate to
        "partly estimated" — neither "estimated" (some priced calls were billed)
        nor silently "provider" (some were not)."""
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'billed.json').write_text(
            json.dumps(self._redteam_payload('B', created='2026-06-29T10:00:00', priced=5, calls=5, estimated=0))
        )
        (rt / 'estimated.json').write_text(
            json.dumps(self._redteam_payload('E', created='2026-06-29T11:00:00', priced=5, calls=5, estimated=5))
        )

        data = metrics.landing([rt, sim])
        assert (data.coverage.priced_calls, data.coverage.estimated_calls) == (10, 5)
        assert '(partly estimated)' in view.landing_body(data)

        ov = metrics.redteam_overview([rt, sim])
        assert (ov.coverage.priced_calls, ov.coverage.estimated_calls) == (10, 5)
        assert '(partly estimated)' in view.redteam_overview_body(ov)

    def test_no_label_when_every_call_priced(self, tmp_path: Path) -> None:
        rt, sim = self._roots(tmp_path, self._redteam_payload('P', created='2026-06-29T10:00:00', priced=10, calls=10))

        assert 'of 10 calls' not in view.landing_body(metrics.landing([rt, sim]))
        assert 'of 10 calls' not in view.redteam_overview_body(metrics.redteam_overview([rt, sim]))

    def test_no_label_for_pre_coverage_reports(self, tmp_path: Path) -> None:
        """Reports saved before priced_calls existed must not be labelled "0 of N"."""
        payload = self._redteam_payload('P', created='2026-06-29T10:00:00', priced=0, calls=10)
        rt, sim = self._roots(tmp_path, payload)

        assert 'calls)' not in view.landing_body(metrics.landing([rt, sim]))
        assert 'calls)' not in view.redteam_overview_body(metrics.redteam_overview([rt, sim]))

    def test_sim_overview_labels_partial_coverage(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (sim / 's.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-30T10:00:00',
                'run_name': 'S',
                'total_results': 2,
                'scorer_averages': {},
                'results': [
                    {
                        'token_usage': {'total_tokens': 100, 'cost_usd': 0.5, 'calls': 1, 'priced_calls': 1},
                        'goal_achieved': True,
                        'turn_count': 1,
                    },
                    {
                        'token_usage': {'total_tokens': 100, 'cost_usd': None, 'calls': 1, 'priced_calls': 0},
                        'goal_achieved': True,
                        'turn_count': 1,
                    },
                ],
            })
        )

        ov = metrics.sim_overview([rt, sim])
        assert (ov.coverage.priced_calls, ov.coverage.cost_calls) == (1, 2)
        assert '(1 of 2 calls)' in view.sim_overview_body(ov)

    def _sim_payload(self, name: str, *, created: str, priced: int, calls: int, estimated: int = 0) -> dict:
        return {
            'mode': 'run',
            'created_at': created,
            'run_name': name,
            'total_results': 1,
            'scorer_averages': {},
            'results': [
                {
                    'token_usage': {
                        'total_tokens': 500,
                        'cost_usd': 0.5,
                        'calls': calls,
                        'priced_calls': priced,
                        'estimated_calls': estimated,
                    },
                    'goal_achieved': True,
                    'turn_count': 1,
                }
            ],
        }

    def test_sim_overview_labels_estimated_coverage(self, tmp_path: Path) -> None:
        """`sim_overview_body` (RES-1022) must carry the same "(estimated)"
        provenance clause the landing and red-team overviews render — a
        regression guard for the ``estimated_calls`` argument `sim_overview_body`
        passes into `_coverage` with no test asserting on it."""
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (sim / 's.json').write_text(
            json.dumps(self._sim_payload('S', created='2026-06-30T10:00:00', priced=10, calls=10, estimated=10))
        )

        ov = metrics.sim_overview([rt, sim])
        assert ov.coverage.priced_calls == ov.coverage.estimated_calls == 10
        assert '(estimated)' in view.sim_overview_body(ov)

    def test_sim_overview_labels_partly_estimated_coverage(self, tmp_path: Path) -> None:
        """Same regression guard as above, for the "(partly estimated)" branch —
        one fully-billed sim run beside one fully-estimated one."""
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (sim / 'billed.json').write_text(
            json.dumps(self._sim_payload('B', created='2026-06-30T10:00:00', priced=5, calls=5, estimated=0))
        )
        (sim / 'estimated.json').write_text(
            json.dumps(self._sim_payload('E', created='2026-06-30T11:00:00', priced=5, calls=5, estimated=5))
        )

        ov = metrics.sim_overview([rt, sim])
        assert (ov.coverage.priced_calls, ov.coverage.estimated_calls) == (10, 5)
        assert '(partly estimated)' in view.sim_overview_body(ov)

    def test_legacy_report_does_not_inflate_the_coverage_denominator(self, tmp_path: Path) -> None:
        """A report predating priced_calls has *unknown* coverage, not zero coverage.

        Counting its calls in the denominator only would report "1 of 11 calls"
        for one new priced call beside ten legacy ones that may all have been
        priced — a fabricated warning, the mirror of the fabricated $0.00 this
        whole feature exists to avoid.
        """
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        legacy = self._redteam_payload('L', created='2026-06-29T10:00:00', priced=0, calls=10)
        del legacy['summary']['token_usage_total']['priced_calls']
        (rt / 'legacy.json').write_text(json.dumps(legacy))
        (rt / 'new.json').write_text(
            json.dumps(self._redteam_payload('N', created='2026-06-29T11:00:00', priced=1, calls=1))
        )

        data = metrics.landing([rt, sim])
        assert (data.coverage.priced_calls, data.coverage.cost_calls, data.coverage.unknown_calls) == (1, 1, 10)
        body = view.landing_body(data)
        assert '1 of 11 calls' not in body
        # ...but the combined $10.50 must not read as authoritative either: 10 of
        # the 11 calls behind it have coverage nobody recorded.
        assert '10 calls of unknown coverage' in body

    def test_legacy_unknown_coverage_beside_fully_estimated_report_keeps_priced_count(
        self, tmp_path: Path
    ) -> None:
        """A pre-``priced_calls`` legacy report (unknown coverage) alongside a
        report whose known calls were entirely catalogue-estimated must show
        *both* the priced count and the provenance — not let provenance displace
        the priced-count clause.

        Regression for the case where ``priced_calls == calls`` (nothing for
        ``cost_coverage`` to qualify on its own) made the "estimated" provenance
        clause the only content, silently dropping how many calls were priced.
        """
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        legacy = self._redteam_payload('L', created='2026-06-29T10:00:00', priced=0, calls=10)
        del legacy['summary']['token_usage_total']['priced_calls']
        (rt / 'legacy.json').write_text(json.dumps(legacy))
        (rt / 'new.json').write_text(
            json.dumps(self._redteam_payload('N', created='2026-06-29T11:00:00', priced=1, calls=1, estimated=1))
        )

        data = metrics.landing([rt, sim])
        assert (
            data.coverage.priced_calls,
            data.coverage.cost_calls,
            data.coverage.unknown_calls,
            data.coverage.estimated_calls,
        ) == (1, 1, 10, 1)
        body = view.landing_body(data)
        assert '1 of 1 calls priced' in body
        assert 'estimated' in body
        assert '10 calls of unknown coverage' in body

    def test_legacy_only_totals_are_labelled_unknown_not_complete(self, tmp_path: Path) -> None:
        """With no new report beside it, a legacy total still has unknown coverage.

        An empty label here would claim every call was priced, which the report
        never said.
        """
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        legacy = self._redteam_payload('L', created='2026-06-29T10:00:00', priced=0, calls=10)
        del legacy['summary']['token_usage_total']['priced_calls']
        (rt / 'legacy.json').write_text(json.dumps(legacy))

        data = metrics.landing([rt, sim])
        assert (data.coverage.priced_calls, data.coverage.cost_calls, data.coverage.unknown_calls) == (0, 0, 10)
        assert '10 calls of unknown coverage' in view.landing_body(data)

    def test_a_legacy_report_with_no_cost_contributes_no_unknown_coverage(self, tmp_path: Path) -> None:
        """Unknown coverage is about cost that was summed. A costless legacy report
        adds nothing to the total, so it has no coverage to be unknown about."""
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        legacy = self._redteam_payload('L', created='2026-06-29T10:00:00', priced=0, calls=10)
        del legacy['summary']['token_usage_total']['priced_calls']
        legacy['summary']['token_usage_total']['cost_usd'] = None
        (rt / 'legacy.json').write_text(json.dumps(legacy))
        (rt / 'new.json').write_text(
            json.dumps(self._redteam_payload('N', created='2026-06-29T11:00:00', priced=1, calls=1))
        )

        data = metrics.landing([rt, sim])
        assert (data.coverage.priced_calls, data.coverage.cost_calls, data.coverage.unknown_calls) == (1, 1, 0)
        assert 'unknown coverage' not in view.landing_body(data)

    def test_no_coverage_label_when_cost_is_unknown(self, tmp_path: Path) -> None:
        """Coverage qualifies a figure that exists — an em dash "no cost" tile with
        "(1 of 2 calls)" under it labels a total that was never shown."""
        payload = self._redteam_payload('P', created='2026-06-29T10:00:00', priced=1, calls=2)
        payload['summary']['token_usage_total']['cost_usd'] = None
        rt, sim = self._roots(tmp_path, payload)

        data = metrics.landing([rt, sim])
        assert data.total_cost is None
        assert '(1 of 2 calls)' not in view.landing_body(data)
        assert '(1 of 2 calls)' not in view.redteam_overview_body(metrics.redteam_overview([rt, sim]))

    def test_present_zero_priced_calls_with_a_cost_is_unknown_not_fully_billed(self, tmp_path: Path) -> None:
        """A custom ``AgentTarget`` can report ``priced_calls: 0`` alongside a
        real ``cost_usd`` (see ``common.target_call._attempt_usage``'s own
        warning for this exact shape). That must land in the same "unknown
        coverage" slot as the legacy no-``priced_calls``-key case, not read as
        "0 of N priced, coverage known" — which would render the cost as fully
        billed with no qualifier at all (RES-1307)."""
        payload = self._redteam_payload('P', created='2026-06-29T10:00:00', priced=0, calls=4)
        rt, sim = self._roots(tmp_path, payload)

        data = metrics.landing([rt, sim])
        assert (data.coverage.priced_calls, data.coverage.cost_calls, data.coverage.unknown_calls) == (0, 0, 4)
        assert data.total_cost is not None
        body = view.landing_body(data)
        assert 'calls priced' not in body
        assert '4 calls of unknown coverage' in body

        ov = metrics.redteam_overview([rt, sim])
        assert (ov.coverage.priced_calls, ov.coverage.cost_calls, ov.coverage.unknown_calls) == (0, 0, 4)
        assert '4 calls of unknown coverage' in view.redteam_overview_body(ov)


class TestScoreTooltip:
    """The Score cell must name what the rate was measured over (RES-1202)."""

    def _row(self, **kw: object) -> metrics.RunRow:
        base: dict = dict(
            id='r', surface='redteam', name='n', when='2026-01-01 00:00', headline='100 attacks',
            score=0.95, status='finished', error=False,
        )
        base.update(kw)
        return metrics.RunRow(**base)

    def test_tooltip_names_the_evaluated_denominator(self) -> None:
        title = view._score_title(self._row(evaluated=60, attacks=100))
        assert 'Resistance rate' in title
        assert '60 of 100 attacks evaluated' in title

    def test_tooltip_and_marker_surface_the_recorded_rate(self) -> None:
        row = self._row(evaluated=60, attacks=100, stored_score=0.71)
        assert '0.71' in view._score_title(row)
        assert 'recalculated' in view._score_title(row)
        assert view._score_marker(row) != ''

    def test_unrecalculated_rows_carry_no_marker(self) -> None:
        assert view._score_marker(self._row(evaluated=100, attacks=100)) == ''

    def test_non_redteam_rows_get_no_denominator(self) -> None:
        title = view._score_title(self._row(surface='sim', evaluated=0, attacks=0))
        assert 'Mean scorer average' in title
        assert 'evaluated' not in title


class TestUnknownSeverity:
    """A vulnerability with no recorded severity gets its own bucket (RES-1202)."""

    def _roots(self, tmp_path: Path, severity: str | None) -> list[Path]:
        rt = tmp_path / 'runs'
        rt.mkdir()
        res = _legacy_result(vulnerable=True)
        if severity is None:
            del res['attack']['severity']
        else:
            res['attack']['severity'] = severity
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload('No severity', created='2026-01-01T00:00:00', resistance=0.0, results=[res])
            )
        )
        return [rt]

    def test_missing_severity_is_not_silently_booked_as_low(self, tmp_path: Path) -> None:
        data = metrics.landing(self._roots(tmp_path, None))
        assert dict(data.severity) == {metrics.UNKNOWN_SEVERITY: 1}

    def test_the_bars_still_sum_to_the_donut(self, tmp_path: Path) -> None:
        # Dropping the bucket instead would leave the severity panel quietly
        # short of the vulnerability count next to it.
        data = metrics.landing(self._roots(tmp_path, None))
        assert sum(n for _, n in data.severity) == data.vulnerable == 1

    def test_unknown_sorts_after_the_real_scale(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        rt.mkdir()
        no_sev = _legacy_result(vulnerable=True)
        del no_sev['attack']['severity']
        (rt / 'legacy_20260101_000000.json').write_text(
            json.dumps(
                _legacy_redteam_payload(
                    'Mixed',
                    created='2026-01-01T00:00:00',
                    resistance=0.0,
                    results=[_legacy_result(vulnerable=True, severity='critical'), no_sev],
                )
            )
        )
        assert [s for s, _ in metrics.landing([rt]).severity] == ['critical', metrics.UNKNOWN_SEVERITY]

    def test_severity_colors_track_the_bucket_not_the_position(self, tmp_path: Path) -> None:
        # Only 'low' survives, so a positional palette would paint it red.
        data = metrics.landing(self._roots(tmp_path, 'low'))
        html = view.landing_body(data)
        assert 'var(--green-600)' in html
        assert 'var(--red-600)' not in html

    def test_off_scale_severity_folds_into_unknown(self, tmp_path: Path) -> None:
        # A present-but-unrecognised value ('sev1') must not create a bucket the
        # display comprehension silently drops. Same failure as the missing
        # field, one step over.
        data = metrics.landing(self._roots(tmp_path, 'sev1'))
        assert dict(data.severity) == {metrics.UNKNOWN_SEVERITY: 1}
        assert sum(n for _, n in data.severity) == data.vulnerable == 1

    def test_off_scale_summary_severity_folds_into_unknown(self) -> None:
        # The stored-summary path has the same display comprehension behind it.
        out = metrics._summary_severity({'by_severity': {'Sev1': {'vulnerabilities_found': 2}, 'HIGH': {'count': 1}}})
        assert out == {metrics.UNKNOWN_SEVERITY: 2, 'high': 1}


class TestLegacyRedTeamCoverage:
    """A legacy red-team report's derived cost carries derived coverage.

    The legacy branch of `metrics.landing` summed ``counts.cost`` into Total
    spend but never touched the coverage counters, so a legacy report's dollars
    entered the total while the qualifier beside them reported complete provider
    billing — the exact case ``unknown_calls`` exists for.
    """

    def _roots(self, tmp_path: Path, results: list[dict]) -> tuple[Path, Path]:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'legacy.json').write_text(
            json.dumps(_legacy_redteam_payload('L', created='2026-06-29T10:00:00', resistance=1.0, results=results))
        )
        return rt, sim

    def test_legacy_derived_cost_counts_as_unknown_coverage(self, tmp_path: Path) -> None:
        rt, sim = self._roots(tmp_path, [_legacy_result(tokens=100), _legacy_result(tokens=200)])

        data = metrics.landing([rt, sim])
        assert data.total_cost is not None
        assert data.coverage.priced_calls == 0
        assert data.coverage.unknown_calls == 2  # one costed usage holder per result
        assert 'unknown coverage' in view.landing_body(data)

    def test_legacy_report_with_no_cost_adds_no_unknown_coverage(self, tmp_path: Path) -> None:
        """Unknown coverage is about cost that was summed; a costless legacy
        report contributes neither dollars nor a qualifier."""
        rt, sim = self._roots(tmp_path, [_legacy_result(), _legacy_result()])

        data = metrics.landing([rt, sim])
        assert data.total_cost is None
        assert (data.coverage.priced_calls, data.coverage.cost_calls, data.coverage.unknown_calls) == (0, 0, 0)
        assert 'unknown coverage' not in view.landing_body(data)


class TestRunGridCostColumn:
    """The per-run Cost cell must carry the same provenance as the KPI band."""

    def _row(self, **kw: object) -> metrics.SimRunRow:
        # Coverage kwargs are collapsed into a `CostCoverage` here so the
        # individual test bodies below can keep passing the four counters by
        # name, matching what they assert about.
        coverage_kw = {k: kw.pop(k) for k in ('priced_calls', 'cost_calls', 'unknown_calls', 'estimated_calls') if k in kw}
        base: dict = dict(
            rid='r', name='n', when=datetime(2026, 6, 29, 10, 0), targets=[('a', 'agent')],
            status='finished', score=0.9, cases=3, cost=0.5, error=False,
            coverage=metrics.CostCoverage(**coverage_kw),
        )
        base.update(kw)
        return metrics.SimRunRow(**base)

    def test_estimated_cost_is_marked_and_explained_in_the_tooltip(self) -> None:
        html = view._run_grid([self._row(priced_calls=2, cost_calls=2, estimated_calls=2)])
        assert '~$0.5000' in html
        assert 'title="estimated"' in html

    def test_partial_coverage_marker_names_the_counts(self) -> None:
        html = view._run_grid([self._row(priced_calls=1, cost_calls=4)])
        assert '~$0.5000' in html
        assert 'title="1 of 4 calls"' in html

    def test_fully_billed_cost_carries_no_marker(self) -> None:
        """``~`` means "qualified" — putting it on every row would say nothing."""
        html = view._run_grid([self._row(priced_calls=4, cost_calls=4)])
        assert '$0.5000' in html
        assert '~' not in html
        assert 'title=' not in html

    def test_unknown_cost_renders_an_em_dash_without_a_marker(self) -> None:
        html = view._run_grid([self._row(cost=None, priced_calls=1, cost_calls=4)])
        assert '—' in html
        assert '~' not in html

    def test_empty_grid_renders_an_empty_state(self) -> None:
        """A head row with nothing under it is indistinguishable from a bug."""
        html = view._run_grid([])
        assert 'runs-empty' in html
        assert 'No runs on this page.' in html


class TestLandingSpendPanelsAreQualified:
    """Avg cost / job and Spend by job type derive from the same data the
    Total spend tile qualifies; neither may render bare dollars."""

    def _roots(self, tmp_path: Path, **usage: object) -> tuple[Path, Path]:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(
            json.dumps({
                'pipeline': {'mode': 'adaptive'},
                'created_at': '2026-06-29T10:00:00',
                'run_name': 'P',
                'total_results': 1,
                'results': [{'attack': {'severity': 'low'}, 'vulnerable': False, 'error': None}],
                'summary': {
                    'resistance_rate': 1.0,
                    'vulnerabilities_found': 0,
                    'evaluated_attacks': 1,
                    'token_usage_total': {'total_tokens': 500, 'cost_usd': 0.5, **usage},
                    'by_severity': {},
                },
            })
        )
        return rt, sim

    def test_avg_cost_tile_inherits_the_totals_coverage(self, tmp_path: Path) -> None:
        rt, sim = self._roots(tmp_path, calls=2, priced_calls=1, estimated_calls=1)
        body = view.landing_body(metrics.landing([rt, sim]))
        # The tile's own sub-line, not just the Total spend one beside it.
        assert (
            '<div class="stat-label">Avg cost / job</div>'
            '<div class="stat-value">$0.5000</div>'
            '<div class="stat-sub">(1 of 2 calls, estimated)</div>'
        ) in body

    def test_spend_by_job_type_does_not_claim_real_cost(self, tmp_path: Path) -> None:
        """``cost_by_kind`` has no per-kind counters, so the panel states the
        limitation instead of asserting the bars are billed dollars."""
        rt, sim = self._roots(tmp_path, calls=2, priced_calls=2, estimated_calls=2)
        body = view.landing_body(metrics.landing([rt, sim]))
        assert 'Real cost across runs' not in body
        assert 'Recorded cost across runs' in body
        assert 'combined coverage (estimated)' in body

    def test_fully_billed_spend_panel_keeps_a_plain_subtitle(self, tmp_path: Path) -> None:
        rt, sim = self._roots(tmp_path, calls=2, priced_calls=2)
        body = view.landing_body(metrics.landing([rt, sim]))
        assert 'Recorded cost across runs' in body
        assert 'combined coverage' not in body


def test_cost_coverage_clamps_priced_calls_to_the_calls_seen() -> None:
    """`priced_calls > cost_calls` reads as fully covered in `coverage_parts`,
    which renders NO qualifier — a malformed report would then present a partly
    priced total as authoritative, the exact defect coverage labels exist to
    prevent. Clamped, not raised: this value travels through the mtime-keyed
    report cache, where one bad report must not take the page down."""
    coverage = metrics.CostCoverage(priced_calls=5, cost_calls=3, estimated_calls=4)
    assert (coverage.priced_calls, coverage.cost_calls, coverage.estimated_calls) == (3, 3, 3)
