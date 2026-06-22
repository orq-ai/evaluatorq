"""TDD tests for dashboard routes: index, report view, export.

Verifies:
- GET /              → 200, contains <section and the report name or 'no reports'
- GET /r/{rid}       → 200, contains <section
- GET /r/{rid}/export → 200, Content-Type text/html, contains <!doctype
- GET /r/missing     → 404
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
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def empty_client(tmp_path: Path) -> TestClient:
    """Client with no reports in the roots."""
    empty = tmp_path / "empty"
    empty.mkdir()
    app = build_app(roots=[empty])
    return TestClient(app, raise_server_exceptions=False)


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

    def test_index_shows_report_cards(self, client: TestClient) -> None:
        r = client.get("/")
        # Both report files should appear; at minimum one report name is present
        text = r.text
        assert "rt_20260101" in text or "sim_20260101" in text or "demo" in text

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

    def test_export_missing_returns_404(self, client: TestClient) -> None:
        r = client.get("/r/nonexistentid123456/export")
        assert r.status_code == 404
