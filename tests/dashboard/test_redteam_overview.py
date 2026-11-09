"""Red Team surface overview: item-level attack metrics, the overview page,
and the real-dollar cost surfaced across the dashboard (RES-1021 / RES-1038)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timedelta
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
        assert 'kpi-band--strip' in r.text

    def test_spend_card_shows_only_the_total(self) -> None:
        from evaluatorq.dashboard import view

        ov = metrics.RedTeamOverview(
            attacks_run=3, break_rate=0.1, critical_findings=0, avg_robustness=0.9, total_cost=1.5, total_input_cost=1.0, total_output_cost=0.5, priced_calls=3, cost_calls=10,
            total_runs=5,
        )
        body = view.redteam_overview_body(ov)
        assert '(in ' not in body
        assert '/ out' not in body
        assert '<div class="kpi-sub">3 of 10 calls priced</div>' in body
        assert '≥' not in body
        assert '<div class="kpi-value">$1.50</div>' in body
        assert '<p class="rt-scope">All 5 runs</p>' in body

    def test_full_coverage_spend_has_no_coverage_line(self) -> None:
        from evaluatorq.dashboard import view

        ov = metrics.RedTeamOverview(
            attacks_run=3, break_rate=0.1, critical_findings=0, avg_robustness=0.9, total_cost=1.5,
            priced_calls=10, cost_calls=10, total_runs=1,
        )
        body = view.redteam_overview_body(ov)
        assert '<div class="kpi-value">$1.50</div>' in body
        assert '<p class="rt-scope">1 run</p>' in body

    def test_kpi_order_and_status_chips(self) -> None:
        from evaluatorq.dashboard import view

        ov = metrics.RedTeamOverview(attacks_run=3, break_rate=0.5, critical_findings=0, avg_robustness=0.5, total_cost=0.0123, total_runs=1)
        body = view.redteam_overview_body(ov)
        labels = [body.index(f'>{n}</div>') for n in ('ASR', 'Critical findings', 'Attacks run', 'Total spend')]
        assert labels == sorted(labels)
        cards = re.findall(r'kpi-card kpi-card--(\w+)"', body)
        assert cards == ['fail', 'pass', 'neutral', 'neutral']
        assert '<div class="kpi-sub">High: above 25%</div>' in body
        for rate, sub in ((0.1, 'At or below 10%'), (0.2, 'Flagged above 10%'), (None, None)):
            ov2 = replace(ov, break_rate=rate)
            html = view.redteam_overview_body(ov2)
            if sub:
                assert f'<div class="kpi-sub">{sub}</div>' in html
            else:
                assert 'kpi-sub' not in html  # no ASR, no sub-line
        assert '$0.0123' in body
        assert 'panel-sub' not in body
        assert '<h2 class="panel-title">Recent runs</h2>' in body

    def test_grid_row_is_labelled_and_pager_is_a_nav(self, tmp_path: Path) -> None:
        from evaluatorq.dashboard import view

        rows = [
            metrics.RedTeamRunRow(
                rid='r1', name='red-team', when=datetime.now() - timedelta(days=3),  # noqa: DTZ005
                targets=[('a', 'agent'), ('b', 'agent'), ('c', 'agent')], status='finished',
                score=0.78, cases=4, cost=0.5, error=False, vulnerable=9, evaluated=40,
            )
        ]
        grid = view._run_grid(rows, redteam=True)
        assert (
            'aria-label="red-team, 3d ago, completed, target a, b, c, ASR 22% (above 10%, flagged), '
            '9/40 broke, 4 cases, cost $0.5000"'
        ) in grid
        assert '<span class="rg-broke-n">9/40</span>' in grid
        assert '>22%<span class="sr-only"> (above 10%, flagged)</span>' in grid
        assert 'rg-bar' not in grid
        assert '<span>Broke</span><span>ASR</span>' in grid
        assert 'completed</span>' not in grid
        assert 'class="run-status ' not in grid
        assert '<span class="target-label">a</span>' in grid
        assert 'class="runs-grid-head" aria-hidden="true"' in grid
        assert '<span class="target-more">+2</span>' in grid
        assert 'title="a, b, c"' in grid
        assert '<span class="rg-name" title="red-team">red-team</span>' in grid
        assert (
            'title="Attack success rate: share of evaluated attacks that broke the agent; '
            f'above {view._ASR_FLAG:.0%} is flagged, above {view._ASR_HIGH:.0%} is high"'
        ) in grid
        single = view._run_grid([replace(rows[0], targets=[('solo-agent', 'agent')])], redteam=True)
        assert '<span class="rg-targets" title="solo-agent">' in single
        assert '<nav class="runs-pager" aria-label="Pagination">' in view._run_pager('redteam', 1, 20, 8)
        assert 'aria-current="true"' in view._run_pager('redteam', 1, 20, 8)

    def test_old_runs_show_absolute_date_and_relative_title(self) -> None:
        from evaluatorq.dashboard import view

        now = datetime.now()  # noqa: DTZ005
        old = now - timedelta(days=30)
        older = datetime(2024, 4, 1, 9, 0)
        rows = [
            metrics.RedTeamRunRow(
                rid=f'r{i}', name='n', when=w, targets=[('a', 'agent')], status='error',
                score=None, cases=1, cost=None, error=False,
            )
            for i, w in enumerate((now - timedelta(days=3), old, older))
        ]
        grid = view._run_grid(rows, redteam=True)
        assert '<span class="rg-sub">3d ago</span>' in grid
        assert f'<span class="rg-sub" title="30d ago">{old:%b} {old.day}' in grid
        assert 'Apr 1, 2024</span>' in grid
        assert '<span class="rg-status"><span class="run-status error">error</span></span>' in grid
        assert 'rg-broke-n' not in grid
        assert 'ASR —, none evaluated,' in grid

    def test_pager_hides_size_picker_when_everything_fits_the_smallest_page(self) -> None:
        from evaluatorq.dashboard import view

        small = view._run_pager('redteam', 1, 3, 8)
        assert 'Per page' not in small
        assert small == ''  # nothing to show: the "All N runs" caption already says the count
        assert view._run_pager('redteam', 1, 8, 8) == ''
        assert 'Per page' in view._run_pager('redteam', 1, 20, 8)
        assert 'Per page' not in view._run_pager('sim', 1, 3, 8)
        assert view._run_pager('sim', 1, 3, 8) == ''
        assert view._run_pager('sim', 1, 1, 8) == ''

    def test_asr_bands_and_boundaries(self) -> None:
        from evaluatorq.dashboard import view

        def row(vulnerable: int, evaluated: int) -> metrics.RedTeamRunRow:
            return metrics.RedTeamRunRow(
                rid='r', name='n', when=datetime.now(), targets=[('a', 'agent')],  # noqa: DTZ005
                status='finished', score=None, cases=evaluated, cost=0.5, error=False,
                vulnerable=vulnerable, evaluated=evaluated,
            )

        def grid(v: int, e: int) -> str:
            return view._run_grid([row(v, e)], redteam=True)

        assert 'run-score rg-score good' in grid(4, 40)  # exactly 10% is fine
        assert 'sr-only' not in grid(4, 40)
        assert 'run-score rg-score mid' in grid(5, 40)  # 12.5%
        assert 'ASR 20% (above 10%, flagged)' in grid(8, 40)
        assert 'ASR 22% (above 10%, flagged)' in grid(9, 40)
        assert 'run-score rg-score mid' in grid(10, 40)  # exactly 25% is still flagged, not high
        assert 'run-score rg-score high' in grid(11, 40)
        assert '(above 25%, high)' in grid(11, 40)
        assert [view._asr_cls(x) for x in (None, 0.0, 0.1, 0.1001, 0.25, 0.2501)] == [
            'none', 'good', 'good', 'mid', 'mid', 'high',
        ]

    def test_kpi_asr_card_uses_the_same_bands(self) -> None:
        from evaluatorq.dashboard import view

        def cards(rate: float | None, critical: int = 0) -> list[str]:
            ov = metrics.RedTeamOverview(
                attacks_run=3, break_rate=rate, critical_findings=critical, avg_robustness=None, total_cost=None, total_runs=1,
            )
            return re.findall(r'kpi-card kpi-card--(\w+)"', view.redteam_overview_body(ov))[:2]

        assert cards(0.1) == ['pass', 'pass']
        assert cards(0.2) == ['warn', 'pass']
        assert cards(0.25, critical=2) == ['warn', 'fail']
        assert cards(0.26) == ['fail', 'pass']
        assert cards(None) == ['neutral', 'pass']

    def test_large_numbers_get_thousands_separators(self) -> None:
        from evaluatorq.dashboard import view

        ov = metrics.RedTeamOverview(
            attacks_run=12345, break_rate=0.5, critical_findings=1234, avg_robustness=None, total_cost=1234567.891,
            priced_calls=1, cost_calls=1, total_runs=1,
            recent=[
                metrics.RedTeamRunRow(
                    rid='r', name='n', when=datetime.now(), targets=[('a', 'agent')],  # noqa: DTZ005
                    status='finished', score=None, cases=12345, cost=1.0, error=False, vulnerable=12345, evaluated=12345,
                )
            ],
        )
        body = view.redteam_overview_body(ov)
        assert '<div class="kpi-value">12,345</div>' in body
        assert '<div class="kpi-value">1,234</div>' in body
        assert '<div class="kpi-value">$1,234,567.89</div>' in body
        assert '<span class="rg-broke-n">12,345/12,345</span>' in body

    def test_no_evaluated_attacks_renders_dashes(self) -> None:
        from evaluatorq.dashboard import view

        run = metrics.RedTeamRunRow(
            rid='r', name='n', when=datetime.now(), targets=[('a', 'agent')],  # noqa: DTZ005
            status='finished', score=None, cases=3, cost=None, error=False,
        )
        grid = view._run_grid([run], redteam=True)
        assert '<span class="rg-broke-n is-none">—</span>' in grid
        assert 'run-score rg-score none" title=' in grid
        assert '>—</span>' in grid

    def test_non_finished_run_keeps_its_status_pill_beside_the_count(self) -> None:
        from evaluatorq.dashboard import view

        run = metrics.RedTeamRunRow(
            rid='r', name='n', when=datetime.now(), targets=[('a', 'agent')],  # noqa: DTZ005
            status='error', score=None, cases=40, cost=None, error=False, vulnerable=9, evaluated=40,
        )
        grid = view._run_grid([run], redteam=True)
        assert '<span class="run-status error">error</span><span class="rg-broke-n">9/40</span>' in grid

    def test_aggregate_fills_vulnerable_and_evaluated(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        rt.mkdir()
        (rt / 'p.json').write_text(
            json.dumps(
                _rt_payload(
                    'P',
                    created='2026-06-29T10:00:00',
                    results=[_attack(severity='low', vulnerable=True), _attack(severity='low', vulnerable=False)],
                    cost=0.001,
                )
            )
        )
        (run,) = metrics.redteam_overview([rt, tmp_path / 'none']).recent
        assert (run.vulnerable, run.evaluated) == (1, 2)

    def test_runs_with_no_attacks_still_render_the_strip_and_grid(self) -> None:
        from evaluatorq.dashboard import view

        run = metrics.RedTeamRunRow(
            rid='r1', name='empty-run', when=datetime.now(), targets=[('a', 'agent')],  # noqa: DTZ005
            status='finished', score=None, cases=0, cost=None, error=False,
        )
        ov = metrics.RedTeamOverview(
            attacks_run=0, break_rate=None, critical_findings=0, avg_robustness=None, total_cost=None,
            recent=[run], total_runs=1,
        )
        body = view.redteam_overview_body(ov)
        assert 'kpi-band--strip' in body
        assert '<div class="kpi-value">—</div>' in body
        assert 'empty-run' in body
        assert 'No reports found' not in body
        none = view.redteam_overview_body(metrics.RedTeamOverview(
            attacks_run=0, break_rate=None, critical_findings=0, avg_robustness=None, total_cost=None,
        ))
        assert 'No reports found' in none


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

    def test_page_past_the_end_shows_the_last_page(self, tmp_path: Path) -> None:
        roots = _seed_many(tmp_path, 20)
        ov = metrics.redteam_overview(roots, page=99, per_page=8)
        assert ov.page == 3
        assert [r.name for r in ov.recent] == [r.name for r in metrics.redteam_overview(roots, page=3).recent]
        assert ov.total_runs == 20
        body = TestClient(build_app(roots=roots)).get('/?surface=redteam&page=99').text
        assert '17&ndash;20 of 20' in body
        assert body.count('class="runs-grid-row"') == 4

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
        assert '<span class="runs-pager-link is-disabled" aria-disabled="true">&lsaquo; Prev</span>' in page1

        last = client.get('/?surface=redteam&page=3').text
        assert '&lsaquo; Prev' in last
        assert '<span class="runs-pager-link is-disabled" aria-disabled="true">Next &rsaquo;</span>' in last


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
        assert 'Total spend' in r.text
        assert '$0.0100' in r.text

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


class TestOutcomeClassification:
    """The evaluation.passed rule on the overview page (declared change, RES-1202).

    The pre-RES-1202 overview loop booked a result whose generation succeeded
    but whose evaluation crashed (error=None, no boolean passed) as resistant.
    The shared classifier drops it from the counts instead, matching
    converters._is_evaluated. These tests pin the overview-page numbers under
    the new rule.
    """

    def test_evaluation_crash_is_dropped_from_overview_counts(self, tmp_path: Path) -> None:
        rt = tmp_path / 'runs'
        rt.mkdir()
        results = [
            {**_attack(severity='low', vulnerable=False), 'evaluation': {'passed': True}},
            {**_attack(severity='critical', vulnerable=True), 'evaluation': {'passed': False}},
            # Judge crashed: no boolean verdict, no target error. Previously
            # counted resistant; now excluded from evaluated entirely.
            {**_attack(severity='high', vulnerable=False), 'evaluation': {'passed': None}},
        ]
        (rt / 'crash_20260101_000000.json').write_text(
            json.dumps(_rt_payload('Judge crash', created='2026-01-01T00:00:00', results=results, cost=0.001))
        )
        ov = metrics.redteam_overview([rt])
        assert ov.attacks_run == 3
        assert ov.break_rate == pytest.approx(1 / 2)
        assert ov.avg_robustness == pytest.approx(1 / 2)
        assert ov.critical_findings == 1

    def test_result_outcome_rules(self) -> None:
        outcome = metrics._result_outcome
        assert outcome({'evaluation': {'passed': True}}) is True
        assert outcome({'evaluation': {'passed': False}}) is False
        assert outcome({'evaluation': {'passed': None}}) is None
        assert outcome({'evaluation': None, 'vulnerable': False}) is None
        assert outcome({'error': 'boom', 'vulnerable': False}) is None
        # Oldest schema: the vulnerable flag is the verdict when present...
        assert outcome({'vulnerable': True}) is False
        assert outcome({'vulnerable': False}) is True
        # ...and its absence means unknown, not resisted.
        assert outcome({}) is None
        assert outcome({'attack': {'severity': 'low'}}) is None
