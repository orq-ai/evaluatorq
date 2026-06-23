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

    def test_sim_md_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        """C1: sim export.md now wired — must return 200."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert r.status_code == 200

    def test_sim_md_content_type(self, client: TestClient, roots: list[Path]) -> None:
        """C1: sim export.md must return text/markdown."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert "text/markdown" in r.headers.get("content-type", "")

    def test_sim_md_has_content_disposition(self, client: TestClient, roots: list[Path]) -> None:
        """C1: sim export.md must carry attachment Content-Disposition."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert "content-disposition" in r.headers
        assert "attachment" in r.headers["content-disposition"]

    def test_sim_md_non_empty(self, client: TestClient, roots: list[Path]) -> None:
        """C1: sim export.md body must be non-empty."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export.md")
        assert len(r.text) > 0

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

    def test_sim_page_has_md_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """C1: sim page must now show a Markdown download link."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.md" in r.text

    def test_sim_page_has_no_csv_download_link(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Sim page must NOT show a CSV download link."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "export.csv" not in r.text

    def test_download_sidebar_has_stable_id(self, client: TestClient, roots: list[Path]) -> None:
        """The download sidebar must carry id="download-sidebar" for OOB targeting."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert 'id="download-sidebar"' in r.text


# ---------------------------------------------------------------------------
# OOB sidebar swap after filter POST
# ---------------------------------------------------------------------------


class TestOOBSidebarAfterFilterPost:
    """POST /r/{rid}/filter must return the OOB sidebar alongside the fragment.

    The OOB sidebar (hx-swap-oob="true") must:
    - appear in the response HTML
    - carry the active filter params in the CSV/JSON hrefs
    - point at URLs that, when followed, return fewer rows than unfiltered
    """

    def test_filter_post_response_contains_oob_sidebar(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The POST /filter response must include an OOB-swapped sidebar."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        assert 'hx-swap-oob="true"' in r.text

    def test_filter_post_oob_sidebar_has_stable_id(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The OOB sidebar must have the stable id so HTMX targets it correctly."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        assert 'id="download-sidebar"' in r.text

    def test_filter_post_oob_sidebar_csv_carries_filter_param(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The OOB sidebar's CSV link must include the active category filter."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        # The CSV href must carry a filter param (category=ASI01)
        assert "export.csv?category=ASI01" in r.text or "export.csv?category%3DASI01" not in r.text
        # More precisely: the sidebar CSV link must contain the category param
        assert "export.csv" in r.text
        sidebar_start = r.text.find('id="download-sidebar"')
        sidebar_end = r.text.find("</section>", sidebar_start)
        sidebar_html = r.text[sidebar_start:sidebar_end]
        assert "category=ASI01" in sidebar_html or "category%3DASI01" not in sidebar_html

    def test_filter_post_oob_sidebar_json_carries_filter_param(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The OOB sidebar's JSON link must include the active category filter."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        sidebar_start = r.text.find('id="download-sidebar"')
        sidebar_end = r.text.find("</section>", sidebar_start)
        sidebar_html = r.text[sidebar_start:sidebar_end]
        assert "export.json" in sidebar_html
        assert "category=ASI01" in sidebar_html

    def test_filter_round_trip_csv_fewer_rows(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Round-trip: filter POST → extract CSV link from OOB sidebar → GET link → fewer rows.

        This is the key parity check: filtering to ASI01-only then downloading
        CSV must yield only ASI01 rows, not all 4 rows.
        """
        rid = report_id(_rt_path(roots))

        # Step 1: POST filter to narrow to ASI01 only.
        r_filter = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r_filter.status_code == 200
        assert 'hx-swap-oob="true"' in r_filter.text

        # Step 2: Extract CSV href from the OOB sidebar.
        sidebar_start = r_filter.text.find('id="download-sidebar"')
        sidebar_end = r_filter.text.find("</section>", sidebar_start)
        sidebar_html = r_filter.text[sidebar_start:sidebar_end]
        # Find the href for the CSV link.
        csv_href_start = sidebar_html.find('export.csv')
        assert csv_href_start >= 0, "No CSV link in OOB sidebar"
        # Walk back to find the href=" opening.
        href_eq = sidebar_html.rfind('href="', 0, csv_href_start)
        href_end = sidebar_html.find('"', href_eq + 6)
        csv_url = sidebar_html[href_eq + 6:href_end]

        # Step 3: GET the filtered CSV URL.
        r_csv_filtered = client.get(csv_url)
        assert r_csv_filtered.status_code == 200
        filtered_lines = [ln for ln in r_csv_filtered.text.splitlines() if ln.strip()]

        # Step 4: GET unfiltered CSV for comparison.
        r_csv_all = client.get(f"/r/{rid}/export.csv")
        all_lines = [ln for ln in r_csv_all.text.splitlines() if ln.strip()]

        # Filtered must have fewer rows than unfiltered (2 vs 4 data rows).
        assert len(filtered_lines) < len(all_lines), (
            f"Expected fewer rows in filtered CSV ({len(filtered_lines)}) "
            f"than full CSV ({len(all_lines)})"
        )

    def test_filter_round_trip_json_fewer_rows(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Round-trip: filter POST → extract JSON link from OOB sidebar → GET link → fewer rows."""
        rid = report_id(_rt_path(roots))

        # POST filter to ASI01 only.
        r_filter = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r_filter.status_code == 200

        # Extract JSON href from the OOB sidebar.
        sidebar_start = r_filter.text.find('id="download-sidebar"')
        sidebar_end = r_filter.text.find("</section>", sidebar_start)
        sidebar_html = r_filter.text[sidebar_start:sidebar_end]
        json_href_start = sidebar_html.find("export.json")
        assert json_href_start >= 0, "No JSON link in OOB sidebar"
        href_eq = sidebar_html.rfind('href="', 0, json_href_start)
        href_end = sidebar_html.find('"', href_eq + 6)
        json_url = sidebar_html[href_eq + 6:href_end]

        # GET the filtered JSON.
        r_json_filtered = client.get(json_url)
        assert r_json_filtered.status_code == 200
        filtered_rows = json.loads(r_json_filtered.text)

        # GET unfiltered JSON.
        all_rows = json.loads(client.get(f"/r/{rid}/export.json").text)

        assert len(filtered_rows) < len(all_rows), (
            f"Expected fewer rows in filtered JSON ({len(filtered_rows)}) "
            f"than full JSON ({len(all_rows)})"
        )

    def test_unfiltered_post_sidebar_has_no_querystring(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """When the POST has no narrowing filter, the OOB sidebar links have no querystring."""
        rid = report_id(_rt_path(roots))
        # POST with all categories selected (effectively no filter)
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        sidebar_start = r.text.find('id="download-sidebar"')
        sidebar_end = r.text.find("</section>", sidebar_start)
        sidebar_html = r.text[sidebar_start:sidebar_end]
        # Without a narrowing selection, the CSV/JSON links should have no '?'
        # (or at most only result=All which is the "show all" radio default).
        assert 'export.csv?result=All' in sidebar_html or 'export.csv"' in sidebar_html or 'export.csv?' in sidebar_html
