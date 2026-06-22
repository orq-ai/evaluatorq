"""TDD tests for dashboard routes: index, report view, export.

Verifies:
- GET /              → 200, contains <section and the report name or 'no reports'
- GET /r/{rid}       → 200, contains <section
- GET /r/{rid}/export → 200, Content-Type text/html, contains <!doctype
- GET /r/missing     → 404
- Malformed-but-sniffable report → non-500 error page (not traceback)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / "runs"
    sim = tmp_path / "sim-runs"
    rt.mkdir()
    sim.mkdir()

    (rt / "rt_20260101_000000.json").write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "pipeline": "static",
                "categories_tested": [],
                "total_results": 0,
                "results": [],
                "summary": {},
            }
        )
    )

    (sim / "sim_20260101_000000.json").write_text(
        json.dumps(
            {
                "run_name": "demo",
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "mode": "run",
                "target_kind": "orq_agent",
                "evaluator_names": [],
                "total_results": 0,
                "scorer_averages": {},
                "results": [],
            }
        )
    )

    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    """Happy-path client — raises on unexpected 500s."""
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def empty_client(tmp_path: Path) -> TestClient:
    """Client with no reports in the roots."""
    empty = tmp_path / "empty"
    empty.mkdir()
    app = build_app(roots=[empty])
    return TestClient(app, raise_server_exceptions=True)


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / "rt_20260101_000000.json"


def _sim_path(roots: list[Path]) -> Path:
    return roots[1] / "sim_20260101_000000.json"


class TestIndexRoute:
    """GET / — report listing."""

    def test_index_returns_200(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200

    def test_index_contains_section_element(self, client: TestClient) -> None:
        r = client.get("/")
        assert "<section" in r.text.lower()

    def test_index_shows_report_cards(self, client: TestClient, roots: list[Path]) -> None:
        r = client.get("/")
        text = r.text
        # Both the redteam report and the sim report must appear.
        assert "rt_20260101" in text
        assert "sim_20260101" in text or "demo" in text

    def test_index_empty_shows_no_reports(self, empty_client: TestClient) -> None:
        r = empty_client.get("/")
        assert r.status_code == 200
        assert "no reports" in r.text.lower()


class TestReportRoute:
    """GET /r/{rid} — embedded report view."""

    def test_report_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200

    def test_report_contains_section_element(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert "<section" in r.text.lower()

    def test_sim_report_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200

    def test_missing_report_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistentid123456")
        assert r.status_code == 404

    def test_malformed_redteam_report_returns_error_page_not_500(
        self, tmp_path: Path
    ) -> None:
        """A file that sniffs as redteam but fails model_validate must render an
        error page (status 200) rather than raising an unhandled 500."""
        rt = tmp_path / "runs"
        rt.mkdir()
        # Has 'pipeline' (sniffs as redteam) but is missing required fields —
        # RedTeamReport.model_validate_json will raise ValidationError.
        (rt / "broken_20260101_000000.json").write_text(
            json.dumps({"pipeline": "static", "results": []})
        )
        broken_app = build_app(roots=[rt])
        # raise_server_exceptions=False so we can inspect the response body
        # rather than having TestClient re-raise the (intentionally handled) error.
        broken_client = TestClient(broken_app, raise_server_exceptions=False)
        rid = report_id(rt / "broken_20260101_000000.json")
        r = broken_client.get(f"/r/{rid}")
        assert r.status_code != 500, f"Expected non-500 but got 500; body: {r.text[:300]}"
        assert "traceback" not in r.text.lower()
        # The error page must contain a visible error indicator.
        assert (
            "report-broken" in r.text
            or "error" in r.text.lower()
        )


class TestExportRoute:
    """GET /r/{rid}/export — standalone HTML export."""

    def test_export_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export")
        assert r.status_code == 200

    def test_export_content_type_is_html(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export")
        assert "text/html" in r.headers.get("content-type", "")

    def test_export_contains_doctype(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/export")
        assert "<!doctype" in r.text.lower()

    def test_sim_export_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export")
        assert r.status_code == 200

    def test_sim_export_contains_doctype(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/export")
        assert "<!doctype" in r.text.lower()

    def test_export_missing_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistentid123456/export")
        assert r.status_code == 404
