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


def _attack(*, severity: str, vulnerable: bool, model: str = "gpt-5.4", error: bool = False) -> dict:
    return {
        "attack": {"severity": severity, "strategy_name": "direct_override"},
        "agent": {"display_name": "Refund agent", "model": model},
        "vulnerable": vulnerable,
        "error": "boom" if error else None,
    }


def _rt_payload(name: str, *, created: str, results: list[dict], cost: float) -> dict:
    evaluated = sum(1 for r in results if not r["error"])
    vulns = sum(1 for r in results if r["vulnerable"] and not r["error"])
    return {
        "pipeline": {"mode": "adaptive"},
        "created_at": created,
        "run_name": name,
        "total_results": len(results),
        "results": results,
        "summary": {
            "resistance_rate": (evaluated - vulns) / evaluated if evaluated else 1.0,
            "vulnerabilities_found": vulns,
            "evaluated_attacks": evaluated,
            "token_usage_total": {"total_tokens": 22650, "cost_usd": cost},
            "by_severity": {},
        },
    }


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / "runs"
    sim = tmp_path / "sim-runs"
    rt.mkdir()
    sim.mkdir()
    (rt / "probe_20260629_101500.json").write_text(
        json.dumps(
            _rt_payload(
                "Probe",
                created="2026-06-29T10:15:00",
                results=[
                    _attack(severity="critical", vulnerable=True),
                    _attack(severity="high", vulnerable=False),
                    _attack(severity="low", vulnerable=False),
                    _attack(severity="medium", vulnerable=False, error=True),
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

    def test_item_rows(self, roots: list[Path]) -> None:
        ov = metrics.redteam_overview(roots)
        assert len(ov.recent) == 4
        rid = report_id(roots[0] / "probe_20260629_101500.json")
        assert all(it.rid == rid for it in ov.recent)
        # The critical vulnerable attack reads as failed (broken through).
        crit = next(it for it in ov.recent if it.severity == "critical")
        assert crit.status == "failed"
        assert crit.model == "gpt-5.4"
        # The errored attack is flagged as warning, not a resist/break.
        errored = next(it for it in ov.recent if it.error)
        assert errored.status == "warning"


class TestRedTeamOverviewScreen:
    def test_page_has_cards_and_table(self, client: TestClient) -> None:
        r = client.get("/?surface=redteam")
        assert r.status_code == 200
        assert "Attacks run" in r.text
        assert "Break rate" in r.text
        assert "Critical findings" in r.text
        assert "Avg robustness" in r.text
        assert "Recent attacks" in r.text
        assert "Refund agent" in r.text


class TestLandingSpend:
    def test_total_cost_and_by_kind(self, tmp_path: Path) -> None:
        rt = tmp_path / "runs"
        sim = tmp_path / "sim-runs"
        rt.mkdir()
        sim.mkdir()
        (rt / "p.json").write_text(
            json.dumps(_rt_payload("P", created="2026-06-29T10:00:00", results=[_attack(severity="low", vulnerable=False)], cost=0.005))
        )
        (sim / "s.json").write_text(
            json.dumps(
                {
                    "mode": "run",
                    "created_at": "2026-06-30T10:00:00",
                    "run_name": "S",
                    "total_results": 1,
                    "scorer_averages": {},
                    "results": [{"token_usage": {"total_tokens": 100, "cost_usd": 0.003}, "goal_achieved": True, "turn_count": 1}],
                }
            )
        )
        data = metrics.landing([rt, sim])
        assert data.total_cost == pytest.approx(0.008)
        by_kind = dict(data.cost_by_kind)
        assert by_kind["Red team"] == pytest.approx(0.005)
        assert by_kind["Agent sim"] == pytest.approx(0.003)

    def test_landing_page_shows_spend(self, tmp_path: Path) -> None:
        rt = tmp_path / "runs"
        sim = tmp_path / "sim-runs"
        rt.mkdir()
        sim.mkdir()
        (rt / "p.json").write_text(
            json.dumps(_rt_payload("P", created="2026-06-29T10:00:00", results=[_attack(severity="low", vulnerable=False)], cost=0.005))
        )
        client = TestClient(build_app(roots=[rt, sim]))
        r = client.get("/")
        assert "Total spend" in r.text
        assert "Spend by job type" in r.text


class TestSimRealCost:
    def test_avg_cost_from_cost_usd(self, tmp_path: Path) -> None:
        sim = tmp_path / "sim-runs"
        rt = tmp_path / "runs"
        sim.mkdir()
        rt.mkdir()
        (sim / "s.json").write_text(
            json.dumps(
                {
                    "mode": "run",
                    "created_at": "2026-06-30T10:00:00",
                    "run_name": "S",
                    "total_results": 2,
                    "scorer_averages": {},
                    "results": [
                        {"token_usage": {"total_tokens": 100, "cost_usd": 0.004}, "goal_achieved": True, "turn_count": 1, "metadata": {"scenario": "a", "persona": "x"}},
                        {"token_usage": {"total_tokens": 200, "cost_usd": 0.006}, "goal_achieved": False, "turn_count": 2, "metadata": {"scenario": "b", "persona": "y"}},
                    ],
                }
            )
        )
        ov = metrics.sim_overview([rt, sim])
        assert ov.avg_cost == pytest.approx(0.005)
        # And the page renders it as a dollar figure, not a token count.
        r = TestClient(build_app(roots=[rt, sim])).get("/?surface=sim")
        assert "Avg cost/sim" in r.text
        assert "$0.00" in r.text
