"""Red Team surface overview: item-level attack metrics, the overview page,
and the real-dollar cost surfaced across the dashboard (RES-1021 / RES-1038)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import metrics
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id


def _attack(*, severity: str, vulnerable: bool, model: str = 'gpt-5.4', error: bool = False) -> dict:
    return {
        'attack': {'severity': severity, 'strategy_name': 'direct_override'},
        'agent': {'display_name': 'Refund agent', 'model': model},
        'vulnerable': vulnerable,
        'error': 'boom' if error else None,
    }


def _rt_payload(name: str, *, created: str, results: list[dict], cost: float) -> dict:
    evaluated = sum(1 for r in results if not r['error'])
    vulns = sum(1 for r in results if r['vulnerable'] and not r['error'])
    return {
        'pipeline': {'mode': 'adaptive'},
        'created_at': created,
        'run_name': name,
        'total_results': len(results),
        'results': results,
        'summary': {
            'resistance_rate': (evaluated - vulns) / evaluated if evaluated else 1.0,
            'vulnerabilities_found': vulns,
            'evaluated_attacks': evaluated,
            'token_usage_total': {'total_tokens': 22650, 'cost_usd': cost},
            'by_severity': {},
        },
    }


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    (rt / 'probe_20260629_101500.json').write_text(
        json.dumps(
            _rt_payload(
                'Probe',
                created='2026-06-29T10:15:00',
                results=[
                    _attack(severity='critical', vulnerable=True),
                    _attack(severity='high', vulnerable=False),
                    _attack(severity='low', vulnerable=False),
                    _attack(severity='medium', vulnerable=False, error=True),
                ],
                cost=0.0048,
            )
        )
    )
    return [rt, sim]


@pytest.fixture
def client(roots: list[Path]) -> TestClient:
    return TestClient(build_app(roots=roots))


class TestRedTeamOverviewMetrics:
    def test_aggregates(self, roots: list[Path]) -> None:
        ov = metrics.redteam_overview(roots)
        assert ov.attacks_run == 4
        # 3 evaluated (one errored), 1 vulnerable → break rate 1/3, robustness 2/3.
        assert ov.break_rate == pytest.approx(1 / 3)
        assert ov.avg_robustness == pytest.approx(2 / 3)
        assert ov.critical_findings == 1
        assert ov.total_cost == pytest.approx(0.0048)

    def test_run_rows(self, roots: list[Path]) -> None:
        ov = metrics.redteam_overview(roots)
        # One row per run (not per attack).
        assert len(ov.recent) == 1
        row = ov.recent[0]
        assert row.rid == report_id(roots[0] / 'probe_20260629_101500.json')
        assert row.name == 'Probe'
        assert row.cases == 4  # four attacks in the run
        # Target is the tested agent surfaced from the run's results.
        assert row.targets == [('Refund agent', 'llm')]  # model, no key → llm chip
        # Score is the run resistance rate (2 of 3 evaluated resisted).
        assert row.score == pytest.approx(2 / 3)
        assert row.cost == pytest.approx(0.0048)


class TestRedTeamOverviewScreen:
    def test_page_has_cards_and_table(self, client: TestClient) -> None:
        r = client.get('/?surface=redteam')
        assert r.status_code == 200
        assert 'Attacks run' in r.text
        assert 'ASR' in r.text  # attack success rate (was 'Break rate')
        assert 'Critical findings' in r.text
        assert 'Avg robustness' not in r.text  # dropped (redundant with ASR)
        assert 'Total spend' in r.text  # replaces the robustness slot
        assert 'Recent runs' in r.text
        assert 'Refund agent' in r.text


def _seed_many(tmp_path: Path, n: int) -> list[Path]:
    """Write ``n`` red team runs (newest last by minute) into a runs root."""
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    for i in range(n):
        (rt / f'probe_{i:03d}.json').write_text(
            json.dumps(
                _rt_payload(
                    f'Probe {i}',
                    created=f'2026-06-29T10:{i:02d}:00',
                    results=[_attack(severity='low', vulnerable=False)],
                    cost=0.001,
                )
            )
        )
    return [rt, sim]


class TestRedTeamOverviewPaging:
    def test_slices_by_page(self, tmp_path: Path) -> None:
        roots = _seed_many(tmp_path, 20)
        first = metrics.redteam_overview(roots, page=1, per_page=8)
        assert first.total_runs == 20
        assert first.page == 1
        assert first.per_page == 8
        assert len(first.recent) == 8
        # Newest first: minute 19 leads page 1.
        assert first.recent[0].name == 'Probe 19'

        third = metrics.redteam_overview(roots, page=3, per_page=8)
        assert len(third.recent) == 4  # 20 - 16
        assert third.recent[0].name == 'Probe 3'

    def test_page_out_of_range_is_empty(self, tmp_path: Path) -> None:
        roots = _seed_many(tmp_path, 20)
        ov = metrics.redteam_overview(roots, page=99, per_page=8)
        assert ov.recent == []
        assert ov.total_runs == 20

    def test_per_page_25_fits_all(self, tmp_path: Path) -> None:
        roots = _seed_many(tmp_path, 20)
        ov = metrics.redteam_overview(roots, page=1, per_page=25)
        assert len(ov.recent) == 20

    def test_per_page_allowlisted(self, tmp_path: Path) -> None:
        roots = _seed_many(tmp_path, 20)
        client = TestClient(build_app(roots=roots))
        # Bad per_page falls back to the default page size (8 rows shown).
        r = client.get('/?surface=redteam&per_page=999')
        assert r.status_code == 200
        assert r.text.count('class="runs-grid-row"') == 8

    def test_pager_nav_links(self, tmp_path: Path) -> None:
        roots = _seed_many(tmp_path, 20)
        client = TestClient(build_app(roots=roots))
        page1 = client.get('/?surface=redteam&page=1').text
        assert 'Next &rsaquo;' in page1
        assert 'href="/?surface=redteam&page=2&per_page=8"' in page1
        # Prev is disabled (a span, not a link) on page 1.
        assert '<span class="runs-pager-link is-disabled">&lsaquo; Prev</span>' in page1

        last = client.get('/?surface=redteam&page=3').text
        assert '&lsaquo; Prev' in last
        assert '<span class="runs-pager-link is-disabled">Next &rsaquo;</span>' in last


class TestLandingSpend:
    def test_total_cost_and_by_kind(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(
            json.dumps(
                _rt_payload(
                    'P', created='2026-06-29T10:00:00', results=[_attack(severity='low', vulnerable=False)], cost=0.005
                )
            )
        )
        (sim / 's.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-30T10:00:00',
                'run_name': 'S',
                'total_results': 1,
                'scorer_averages': {},
                'results': [
                    {'token_usage': {'total_tokens': 100, 'cost_usd': 0.003}, 'goal_achieved': True, 'turn_count': 1}
                ],
            })
        )
        data = metrics.landing([rt, sim])
        assert data.total_cost == pytest.approx(0.008)
        by_kind = dict(data.cost_by_kind)
        assert by_kind['Red team'] == pytest.approx(0.005)
        assert by_kind['Agent sim'] == pytest.approx(0.003)

    def test_landing_page_shows_spend(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (rt / 'p.json').write_text(
            json.dumps(
                _rt_payload(
                    'P', created='2026-06-29T10:00:00', results=[_attack(severity='low', vulnerable=False)], cost=0.005
                )
            )
        )
        client = TestClient(build_app(roots=[rt, sim]))
        r = client.get('/')
        assert 'Total spend' in r.text
        assert 'Spend by job type' in r.text


class TestSimRealCost:
    def test_avg_cost_from_cost_usd(self, tmp_path: Path) -> None:
        sim = tmp_path / 'sim-runs'
        rt = tmp_path / 'runs'
        sim.mkdir()
        rt.mkdir()
        (sim / 's.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-30T10:00:00',
                'run_name': 'S',
                'total_results': 2,
                'scorer_averages': {},
                'results': [
                    {
                        'token_usage': {'total_tokens': 100, 'cost_usd': 0.004},
                        'goal_achieved': True,
                        'turn_count': 1,
                        'metadata': {'scenario': 'a', 'persona': 'x'},
                    },
                    {
                        'token_usage': {'total_tokens': 200, 'cost_usd': 0.006},
                        'goal_achieved': False,
                        'turn_count': 2,
                        'metadata': {'scenario': 'b', 'persona': 'y'},
                    },
                ],
            })
        )
        ov = metrics.sim_overview([rt, sim])
        assert ov.avg_cost == pytest.approx(0.005)
        # And the page renders it as a dollar figure, not a token count.
        r = TestClient(build_app(roots=[rt, sim])).get('/?surface=sim')
        assert 'Avg cost/sim' in r.text
        assert '$0.00' in r.text

    def test_avg_cost_ignores_costless_sims(self, tmp_path: Path) -> None:
        """Mixed store: a sim without cost_usd is 'unknown', not $0 — it must not
        dilute the mean. avg_cost divides by costed sims, not all sims."""
        sim = tmp_path / 'sim-runs'
        rt = tmp_path / 'runs'
        sim.mkdir()
        rt.mkdir()
        (sim / 's.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-30T10:00:00',
                'run_name': 'S',
                'total_results': 2,
                'scorer_averages': {},
                'results': [
                    {'token_usage': {'total_tokens': 100, 'cost_usd': 0.006}, 'goal_achieved': True, 'turn_count': 1},
                    # Older sim recorded before cost_usd existed — no cost key.
                    {'token_usage': {'total_tokens': 200}, 'goal_achieved': False, 'turn_count': 2},
                ],
            })
        )
        ov = metrics.sim_overview([rt, sim])
        # One costed sim at 0.006 → 0.006, not 0.003 (would be the /total bug).
        assert ov.avg_cost == pytest.approx(0.006)


class TestZeroEvaluatedNoScore:
    """A zero-evaluated run must show no resistance score anywhere the rate
    renders: landing rows (covered in test_landing), the red-team overview
    rows, and the detail KPI band."""

    @pytest.fixture
    def zero_eval_roots(self, tmp_path: Path) -> tuple[list[Path], Path]:
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        report = rt / 'empty_20260731_130000.json'
        report.write_text(
            json.dumps(
                _rt_payload(
                    'Empty run',
                    created='2026-07-31T13:00:00',
                    # Every attack errored: evaluated_attacks == 0 and the
                    # payload builder emits the poisoned default rate of 1.0.
                    results=[_attack(severity='low', vulnerable=False, error=True)],
                    cost=0.0,
                )
            )
        )
        return [rt, sim], report

    def test_overview_row_has_no_score(self, zero_eval_roots: tuple[list[Path], Path]) -> None:
        roots, _report = zero_eval_roots
        ov = metrics.redteam_overview(roots)
        assert len(ov.recent) == 1
        assert ov.recent[0].score is None

    def test_overview_route_never_renders_perfect_score(self, zero_eval_roots: tuple[list[Path], Path]) -> None:
        roots, _report = zero_eval_roots
        client = TestClient(build_app(roots=roots))
        r = client.get('/?surface=redteam')
        assert r.status_code == 200
        # The score cell renders the no-score em dash, never the poisoned 1.00.
        assert '1.00' not in r.text
        assert '>\u2014<' in r.text
