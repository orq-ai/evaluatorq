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

    def test_item_rows(self, roots: list[Path]) -> None:
        ov = metrics.sim_overview(roots)
        assert len(ov.recent) == 3
        first = ov.recent[0]
        assert first.scenario == 'billing'
        assert first.persona == 'alice'
        assert first.model == 'gpt-5.4'
        assert first.turns == 4
        assert first.outcome == 'passed'  # score >= 0.8
        # Every row links to a real report id.
        rid = report_id(roots[1] / 'support_20260625_140000.json')
        assert all(it.rid == rid for it in ov.recent)
        # The errored sim is flagged and reads as failed.
        errored = next(it for it in ov.recent if it.error)
        assert errored.outcome == 'failed'

    def test_empty(self, tmp_path: Path) -> None:
        empty = [tmp_path / 'runs', tmp_path / 'sim-runs']
        for p in empty:
            p.mkdir()
        ov = metrics.sim_overview(empty)
        assert ov.simulations_run == 0
        assert ov.goal_completion is None
        assert ov.recent == []


class TestSimOverviewScreen:
    def test_page_has_cards_and_table(self, client: TestClient) -> None:
        r = client.get('/?surface=sim')
        assert r.status_code == 200
        assert 'Simulations run' in r.text
        assert 'Goal completion' in r.text
        assert 'Avg turns' in r.text
        assert 'Avg tokens/sim' in r.text  # cost stand-in
        assert 'Recent simulations' in r.text
        # Item-level rows surface the scenario/persona, and link to the report.
        assert 'billing' in r.text
        assert 'alice' in r.text

    def test_empty_surface(self, tmp_path: Path) -> None:
        empty = [tmp_path / 'runs', tmp_path / 'sim-runs']
        for p in empty:
            p.mkdir()
        c = TestClient(build_app(roots=empty))
        r = c.get('/?surface=sim')
        assert r.status_code == 200
        assert 'no reports' in r.text.lower()


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
