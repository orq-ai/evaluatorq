"""Tests for the combined Dashboard landing, per-kind run lists, and the
metrics aggregation that feeds them (RES-974)."""

from __future__ import annotations

import json
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

