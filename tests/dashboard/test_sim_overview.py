"""Agent Sim surface overview: item-level metrics, the rich overview page, and
the outcomes donut on the report Overview tab (RES-1022)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import metrics
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id


def _result(
    *,
    persona: str,
    scenario: str,
    model: str,
    goal: bool,
    score: float,
    turns: int,
    tokens: int,
    terminated_by: str = 'judge',
) -> dict:
    return {
        'terminated_by': terminated_by,
        'goal_achieved': goal,
        'goal_completion_score': score,
        'turn_count': turns,
        'total_tokens': tokens,
        'metadata': {'persona': persona, 'scenario': scenario, 'model': model},
    }


def _sim_payload(name: str, *, created: str, results: list[dict]) -> dict:
    return {
        'mode': 'run',
        'created_at': created,
        'run_name': name,
        'total_results': len(results),
        'scorer_averages': {'goal_achieved': 0.5},
        'results': results,
    }


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    (sim / 'support_20260625_140000.json').write_text(
        json.dumps(
            _sim_payload(
                'Support sim',
                created='2026-06-25T14:00:00',
                results=[
                    _result(
                        persona='alice',
                        scenario='billing',
                        model='gpt-5.4',
                        goal=True,
                        score=0.95,
                        turns=4,
                        tokens=1200,
                    ),
                    _result(
                        persona='bob', scenario='refunds', model='gpt-5.4', goal=False, score=0.4, turns=6, tokens=1800
                    ),
                    _result(
                        persona='carol',
                        scenario='signup',
                        model='gpt-5.4',
                        goal=False,
                        score=0.0,
                        turns=2,
                        tokens=500,
                        terminated_by='error',
                    ),
                ],
            )
        )
    )
    return [rt, sim]


@pytest.fixture
def client(roots: list[Path]) -> TestClient:
    return TestClient(build_app(roots=roots))


class TestSimOverviewMetrics:
    def test_aggregates(self, roots: list[Path]) -> None:
        ov = metrics.sim_overview(roots)
        assert ov.simulations_run == 3
        # One of three achieved the goal.
        assert ov.goal_completion == pytest.approx(1 / 3)
        assert ov.avg_turns == pytest.approx((4 + 6 + 2) / 3)
        assert ov.avg_tokens == pytest.approx((1200 + 1800 + 500) / 3)
        # Outcomes split for the donut.
        assert (ov.achieved, ov.not_achieved, ov.errors) == (1, 1, 1)

    def test_run_rows(self, roots: list[Path]) -> None:
        ov = metrics.sim_overview(roots)
        # One row per run (not per simulation case).
        assert len(ov.recent) == 1
        row = ov.recent[0]
        assert row.name == 'Support sim'
        assert row.cases == 3  # three simulations in the run
        # Score is the mean of scorer_averages ({'goal_achieved': 0.5}).
        assert row.score == pytest.approx(0.5)
        assert row.status == 'finished'  # lifecycle — completed run (score carries quality)
        # Row links to the run's report id.
        assert row.rid == report_id(roots[1] / 'support_20260625_140000.json')

    def test_target_name_and_model(self, tmp_path: Path) -> None:
        """A run persisting `target` + `target_model` shows 'name · model'; a run
        where every case errored reads Status = 'error'."""
        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        (sim / 'named.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-30T10:00:00',
                'run_name': 'sim:x',
                'target_kind': 'orq_agent',
                'target': 'support-bot',
                'target_model': 'gpt-4o',
                'total_results': 1,
                'scorer_averages': {'goal_achieved': 1.0},
                'results': [{'goal_achieved': True, 'turn_count': 1, 'metadata': {}}],
            })
        )
        (sim / 'broken.json').write_text(
            json.dumps({
                'mode': 'run',
                'created_at': '2026-06-29T10:00:00',
                'run_name': 'sim:y',
                'target_kind': 'orq_agent',
                'target': 'refund-bot',
                'total_results': 2,
                'scorer_averages': {},
                'results': [
                    {'terminated_by': 'error', 'metadata': {}},
                    {'terminated_by': 'error', 'metadata': {}},
                ],
            })
        )
        rows = {r.name: r for r in metrics.sim_overview([rt, sim]).recent}
        # Agent target with a known model → "name · model".
        assert rows['sim:x'].targets == [('support-bot · gpt-4o', 'agent')]
        # agent target name surfaces (never a generic 'Orq agent' label).
        assert rows['sim:y'].targets == [('refund-bot', 'agent')]
        # Every case errored → lifecycle status 'error'.
        assert rows['sim:y'].status == 'error'

    def test_empty(self, tmp_path: Path) -> None:
        empty = [tmp_path / 'runs', tmp_path / 'sim-runs']
        for p in empty:
            p.mkdir()
        ov = metrics.sim_overview(empty)
        assert ov.simulations_run == 0
        assert ov.goal_completion is None
        assert ov.recent == []


def _seed_many_sim(tmp_path: Path, n: int) -> list[Path]:
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    for i in range(n):
        (sim / f'sim_{i:03d}.json').write_text(
            json.dumps(
                _sim_payload(
                    f'Sim {i}',
                    created=f'2026-06-25T14:{i:02d}:00',
                    results=[
                        _result(
                            persona='alice',
                            scenario='billing',
                            model='gpt-5.4',
                            goal=True,
                            score=0.9,
                            turns=3,
                            tokens=900,
                        )
                    ],
                )
            )
        )
    return [rt, sim]


class TestSimOverviewPaging:
    def test_slices_by_page(self, tmp_path: Path) -> None:
        roots = _seed_many_sim(tmp_path, 20)
        first = metrics.sim_overview(roots, page=1, per_page=8)
        assert first.total_runs == 20
        assert first.page == 1
        assert first.per_page == 8
        assert len(first.recent) == 8
        assert first.recent[0].name == 'Sim 19'  # newest first

        third = metrics.sim_overview(roots, page=3, per_page=8)
        assert len(third.recent) == 4
        assert third.recent[0].name == 'Sim 3'

    def test_page_out_of_range_is_empty(self, tmp_path: Path) -> None:
        roots = _seed_many_sim(tmp_path, 20)
        ov = metrics.sim_overview(roots, page=99, per_page=8)
        assert ov.recent == []
        assert ov.total_runs == 20

    def test_pager_nav_links(self, tmp_path: Path) -> None:
        roots = _seed_many_sim(tmp_path, 20)
        client = TestClient(build_app(roots=roots))
        page1 = client.get('/?surface=sim&page=1').text
        assert 'Next &rsaquo;' in page1
        assert 'href="/?surface=sim&page=2&per_page=8"' in page1


class TestSimOverviewScreen:
    def test_page_has_cards_and_table(self, client: TestClient) -> None:
        r = client.get('/?surface=sim')
        assert r.status_code == 200
        assert 'Simulations run' in r.text
        assert 'Goal completion' in r.text
        assert 'Avg turns' in r.text
        assert 'Avg tokens/sim' in r.text  # cost stand-in
        assert 'Recent runs' in r.text
        # Run-level rows surface the run name + design columns.
        assert 'Support sim' in r.text
        assert 'Cases' in r.text

    def test_empty_surface(self, tmp_path: Path) -> None:
        empty = [tmp_path / 'runs', tmp_path / 'sim-runs']
        for p in empty:
            p.mkdir()
        c = TestClient(build_app(roots=empty))
        r = c.get('/?surface=sim')
        assert r.status_code == 200
        assert 'no reports' in r.text.lower()


def test_target_pill_normalizes_persisted_kind_and_marks_external() -> None:
    """Run-list target icons come only from saved JSON target kinds."""
    from evaluatorq.dashboard.view import _target_pill

    assert 'data-kind="agent"' in _target_pill('flight-delay-analyst', 'orq_agent')
    external = _target_pill('tailscale-openai', 'callback')
    assert 'data-kind="external"' in external
    assert '<span class="dot"></span>' not in external


class TestOutcomesDonut:
    def test_donut_on_report_overview(self, tmp_path: Path) -> None:
        from tests.dashboard.test_downloads import _make_sim_run

        rt = tmp_path / 'runs'
        sim = tmp_path / 'sim-runs'
        rt.mkdir()
        sim.mkdir()
        sim_file = sim / 'sim.json'
        sim_file.write_text(
            _make_sim_run(personas=['alice', 'bob'], goal_achieved_flags=[True, False]).model_dump_json()
        )

        client = TestClient(build_app(roots=[rt, sim]))
        rid = report_id(sim_file)
        html = client.get(f'/r/{rid}').text
        assert 'Outcomes' in html
        assert 'donut-legend' in html
        assert 'Achieved' in html
