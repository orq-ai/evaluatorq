"""TDD tests for download routes: export.html, export.md, export.csv, export.json.

Verifies:
- GET /r/{rid}/export.html  → 200, text/html, Content-Disposition attachment
- GET /r/{rid}/export.md    → 200 redteam + text/markdown + Content-Disposition;
                              404 for sim (honest parity — sim never had md export)
- GET /r/{rid}/export.csv   → 200 redteam + text/csv + Content-Disposition;
                              404 for sim (sim never had CSV export)
- GET /r/{rid}/export.json  → 200 both surfaces + application/json + Content-Disposition
- csv/json honor filter params: a filter that drops rows yields fewer rows
- missing rid → 404 for all download routes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id

# ---------------------------------------------------------------------------
# Shared fixture factories
# ---------------------------------------------------------------------------


def _make_rt_report():
    """Build a minimal RedTeamReport with results in two categories."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.redteam.contracts import (
        AgentInfo,
        AttackInfo,
        AttackTechnique,
        DeliveryMethod,
        Framework,
        Pipeline,
        RedTeamReport,
        RedTeamResult,
        Severity,
        TurnType,
        UnifiedEvaluationResult,
    )
    from evaluatorq.redteam.reports.converters import compute_report_summary

    def _result(
        category: str,
        passed: bool,
        agent_key: str = "agent-a",
        attack_id: str | None = None,
    ) -> RedTeamResult:
        aid = attack_id or f"{category}-{agent_key}-{passed}"
        return RedTeamResult(
            attack=AttackInfo(
                id=aid,
                category=category,
                vulnerability="",
                framework=Framework.OWASP_ASI,
                attack_technique=AttackTechnique.INDIRECT_INJECTION,
                delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
                turn_type=TurnType.SINGLE,
                severity=Severity.MEDIUM,
                source="test",
            ),
            agent=AgentInfo(key=agent_key),
            messages=[],
            vulnerable=not passed,
            response="ok",
            evaluation=UnifiedEvaluationResult(passed=passed, explanation="test"),
        )

    results = [
        _result("ASI01", passed=True),
        _result("ASI01", passed=False),
        _result("LLM01", passed=False),
        _result("LLM01", passed=True),
    ]
    summary = compute_report_summary(results)
    return RedTeamReport(
        pipeline=Pipeline.STATIC,
        created_at=datetime.now(tz=timezone.utc),
        categories_tested=["ASI01", "LLM01"],
        total_results=len(results),
        results=results,
        summary=summary,
        description="test redteam report",
    )


def _make_sim_run(
    personas: list[str] | None = None,
    goal_achieved_flags: list[bool] | None = None,
):
    """Build a minimal SimulationRun."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy

    personas = personas or ["alice", "bob"]
    goal_achieved_flags = goal_achieved_flags or [True, False]

    results = []
    for persona, goal in zip(personas, goal_achieved_flags):
        results.append(
            SimulationResult(
                messages=[],
                terminated_by=TerminatedBy.judge,
                reason="done",
                goal_achieved=goal,
                goal_completion_score=1.0 if goal else 0.0,
                rules_broken=[],
                turn_count=2,
                turn_metrics=[],
                token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                metadata={"persona": persona, "scenario": "billing"},
            )
        )
    return SimulationRun(
        run_name="test-sim",
        created_at=datetime.now(tz=timezone.utc),
        mode="run",
        target_kind="orq_agent",
        evaluator_names=["goal_achieved"],
        total_results=len(results),
        scorer_averages={"goal_achieved": 0.5},
        results=results,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / "runs"
    sim = tmp_path / "sim-runs"
    rt.mkdir()
    sim.mkdir()

    rt_path = rt / "rt_dl_test.json"
    rt_path.write_text(_make_rt_report().model_dump_json())

    sim_path = sim / "sim_dl_test.json"
    sim_path.write_text(_make_sim_run().model_dump_json())

    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / "rt_dl_test.json"


def _sim_path(roots: list[Path]) -> Path:
    return roots[1] / "sim_dl_test.json"


# ---------------------------------------------------------------------------
# export.html
# ---------------------------------------------------------------------------


class TestExportHtml:
    def test_redteam_html_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.html")
        assert r.status_code == 200

    def test_redteam_html_content_type(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.html")
        assert "text/html" in r.headers.get("content-type", "")

    def test_redteam_html_non_empty(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.html")
        assert len(r.text) > 0

    def test_redteam_html_has_content_disposition(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.html")
        assert "content-disposition" in r.headers

    def test_sim_html_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.html")
        assert r.status_code == 200

    def test_missing_html_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistent123/export.html")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# export.md
# ---------------------------------------------------------------------------


class TestExportMd:
    def test_redteam_md_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert r.status_code == 200

    def test_redteam_md_content_type(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert "text/markdown" in r.headers.get("content-type", "")

    def test_redteam_md_non_empty(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert len(r.text) > 0

    def test_redteam_md_has_content_disposition(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert "content-disposition" in r.headers
        assert "attachment" in r.headers["content-disposition"]

    def test_sim_md_returns_404(self, client: TestClient, roots: list[Path]) -> None:
        """Sim never had a Markdown export — honest parity, must return 404."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert r.status_code == 404

    def test_sim_md_body_explains_reason(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert "simulation" in r.text.lower() or "markdown" in r.text.lower()

    def test_missing_md_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistent123/export.md")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# export.csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_redteam_csv_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.csv")
        assert r.status_code == 200

    def test_redteam_csv_content_type(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.csv")
        assert "text/csv" in r.headers.get("content-type", "")

    def test_redteam_csv_non_empty(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.csv")
        assert len(r.text) > 0

    def test_redteam_csv_has_content_disposition(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.csv")
        assert "content-disposition" in r.headers
        assert "attachment" in r.headers["content-disposition"]

    def test_redteam_csv_has_header_row(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.csv")
        first_line = r.text.splitlines()[0]
        assert "ID" in first_line or "Category" in first_line

    def test_redteam_csv_filter_reduces_rows(self, client: TestClient, roots: list[Path]) -> None:
        """Applying a category filter should yield fewer data rows than unfiltered."""
        rid = report_id(_rt_path(roots))
        # Full export: 4 results → 4 data rows + 1 header
        r_all = client.get(f"/r/{rid}/export.csv")
        all_lines = [l for l in r_all.text.splitlines() if l.strip()]

        # Filter to only ASI01 → 2 results
        r_filtered = client.get(f"/r/{rid}/export.csv?category=ASI01")
        filtered_lines = [l for l in r_filtered.text.splitlines() if l.strip()]

        assert len(filtered_lines) < len(all_lines)

    def test_sim_csv_returns_404(self, client: TestClient, roots: list[Path]) -> None:
        """Sim never had a CSV export — honest parity."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.csv")
        assert r.status_code == 404

    def test_missing_csv_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistent123/export.csv")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# export.json
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_redteam_json_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert r.status_code == 200

    def test_redteam_json_content_type(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert "application/json" in r.headers.get("content-type", "")

    def test_redteam_json_non_empty(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert len(r.text) > 0

    def test_redteam_json_has_content_disposition(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert "content-disposition" in r.headers
        assert "attachment" in r.headers["content-disposition"]

    def test_redteam_json_is_valid_json(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        data = json.loads(r.text)
        assert isinstance(data, list)

    def test_redteam_json_filter_reduces_rows(self, client: TestClient, roots: list[Path]) -> None:
        """Applying a category filter should yield fewer rows than unfiltered."""
        rid = report_id(_rt_path(roots))
        all_data = json.loads(client.get(f"/r/{rid}/export.json").text)
        filtered_data = json.loads(client.get(f"/r/{rid}/export.json?category=ASI01").text)
        assert len(filtered_data) < len(all_data)

    def test_sim_json_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert r.status_code == 200

    def test_sim_json_content_type(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert "application/json" in r.headers.get("content-type", "")

    def test_sim_json_non_empty(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert len(r.text) > 0

    def test_sim_json_has_content_disposition(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        assert "content-disposition" in r.headers
        assert "attachment" in r.headers["content-disposition"]

    def test_sim_json_is_valid_json(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.json")
        data = json.loads(r.text)
        assert isinstance(data, list)

    def test_sim_json_filter_reduces_rows(self, client: TestClient, roots: list[Path]) -> None:
        """Applying a persona filter to sim should yield fewer rows than unfiltered."""
        rid = report_id(_sim_path(roots))
        all_data = json.loads(client.get(f"/r/{rid}/export.json").text)
        filtered_data = json.loads(client.get(f"/r/{rid}/export.json?persona=alice").text)
        assert len(filtered_data) < len(all_data)

    def test_missing_json_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistent123/export.json")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Download sidebar appears on report page
# ---------------------------------------------------------------------------


class TestDownloadSidebarOnReportPage:
    def test_redteam_page_has_html_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.html" in r.text

    def test_redteam_page_has_md_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.md" in r.text

    def test_redteam_page_has_csv_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.csv" in r.text

    def test_redteam_page_has_json_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.json" in r.text

    def test_sim_page_has_html_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.html" in r.text

    def test_sim_page_has_json_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.json" in r.text

    def test_sim_page_has_no_md_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Sim page must NOT show a Markdown download link."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        # The download sidebar should not link to export.md for sim reports.
        assert "export.md" not in r.text

    def test_sim_page_has_no_csv_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Sim page must NOT show a CSV download link."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.csv" not in r.text
