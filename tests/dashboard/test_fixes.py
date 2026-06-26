"""Tests for the four targeted dashboard fixes.

Fix 1: eq ui FILE prints direct report URL
Fix 2: GET /?surface=sim shows only sim cards
Fix 3: read_json cache hits on second load of unchanged file
Fix 4: filter apply() runs once per POST (recompute_options takes filtered list)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import read_json, report_id


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _redteam_payload() -> dict:
    return {
        "version": "2.0.0",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "pipeline": "static",
        "categories_tested": [],
        "total_results": 0,
        "results": [],
        "summary": {},
    }


def _sim_payload() -> dict:
    return {
        "run_name": "demo-sim",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "mode": "run",
        "target_kind": "orq_agent",
        "evaluator_names": [],
        "total_results": 0,
        "scorer_averages": {},
        "results": [],
    }


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / "runs"
    sim = tmp_path / "sim-runs"
    rt.mkdir()
    sim.mkdir()
    (rt / "rt_fix_test.json").write_text(json.dumps(_redteam_payload()))
    (sim / "sim_fix_test.json").write_text(json.dumps(_sim_payload()))
    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Fix 1: CLI prints direct URL when a file path is passed
# ---------------------------------------------------------------------------


class TestFix1CliDirectUrl:
    """Fix 1 — eq ui FILE surfaces the direct report URL via typer.echo.

    The top-level ``cli.app`` has ``ui`` as its *only* registered command and
    is therefore itself the ``ui`` callback — typer exposes it without a
    subcommand word.  Invoke as ``runner.invoke(app, [str(path)])`` (no "ui"
    prefix).
    """

    def test_ui_file_path_prints_direct_url(self, tmp_path: Path) -> None:
        """When a file is passed, the CLI must print the /r/{rid} URL."""
        from typer.testing import CliRunner

        from evaluatorq.cli import app as cli_app

        rt_dir = tmp_path / "runs"
        rt_dir.mkdir()
        report_file = rt_dir / "rt_20260101_000000.json"
        report_file.write_text(json.dumps(_redteam_payload()))

        rid = report_id(report_file)
        runner = CliRunner()

        # Invoke without actually starting uvicorn — patch serve to return immediately.
        with patch("evaluatorq.dashboard.launch.serve"):
            result = runner.invoke(cli_app, [str(report_file)])

        assert result.exit_code == 0, result.output
        # The output must contain the direct report URL.
        assert f"/r/{rid}" in result.output

    def test_ui_file_path_scans_parent_dir(self, tmp_path: Path) -> None:
        """roots passed to serve() must be the file's parent directory."""
        from typer.testing import CliRunner

        from evaluatorq.cli import app as cli_app

        rt_dir = tmp_path / "runs"
        rt_dir.mkdir()
        report_file = rt_dir / "rt_20260101_000000.json"
        report_file.write_text(json.dumps(_redteam_payload()))

        runner = CliRunner()
        captured_roots: list | None = None

        def _fake_serve(roots: list, *, host: str, port: int) -> None:  # noqa: ARG001
            nonlocal captured_roots
            captured_roots = roots

        with patch("evaluatorq.dashboard.launch.serve", side_effect=_fake_serve):
            runner.invoke(cli_app, [str(report_file)])

        assert captured_roots is not None
        assert captured_roots == [rt_dir]

    def test_ui_dir_path_does_not_print_direct_url(self, tmp_path: Path) -> None:
        """When a directory is passed, no /r/{rid} URL should appear."""
        from typer.testing import CliRunner

        from evaluatorq.cli import app as cli_app

        rt_dir = tmp_path / "runs"
        rt_dir.mkdir()

        runner = CliRunner()
        with patch("evaluatorq.dashboard.launch.serve"):
            result = runner.invoke(cli_app, [str(rt_dir)])

        assert "/r/" not in result.output


# ---------------------------------------------------------------------------
# Fix 2: GET /?surface=sim shows only sim cards
# ---------------------------------------------------------------------------


class TestFix2SurfaceNavFilter:
    """Fix 2 — ?surface query param filters the index page to one surface."""

    def test_surface_sim_shows_only_sim_cards(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /?surface=sim must show the sim report and exclude the redteam one."""
        r = client.get("/?surface=sim")
        assert r.status_code == 200
        # Sim report name appears.
        assert "demo-sim" in r.text
        # Redteam report must not appear (its stem is rt_fix_test).
        assert "rt_fix_test" not in r.text

    def test_surface_redteam_shows_only_redteam_cards(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /?surface=redteam must show the redteam report and exclude the sim one."""
        r = client.get("/?surface=redteam")
        assert r.status_code == 200
        # Redteam report stem appears.
        assert "rt_fix_test" in r.text or "Red Team" in r.text
        # Sim report name must not appear.
        assert "demo-sim" not in r.text

    def test_no_surface_param_shows_all_cards(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET / with no surface param must show both surfaces."""
        r = client.get("/")
        assert r.status_code == 200
        assert "demo-sim" in r.text

    def test_surface_nav_link_active_class_matches(
        self, client: TestClient
    ) -> None:
        """The sim nav link must carry class='active' when ?surface=sim is requested."""
        r = client.get("/?surface=sim")
        assert r.status_code == 200
        # The active sidebar nav item carries the active class and is the
        # Agent Sim link (/?surface=sim).
        assert "nav-item active" in r.text
        assert '<a class="nav-item active" href="/?surface=sim"' in r.text

    def test_nav_links_use_surface_hrefs(self, client: TestClient) -> None:
        """Nav links must point to /?surface=... not just /."""
        r = client.get("/")
        assert "/?surface=redteam" in r.text
        assert "/?surface=sim" in r.text

    def test_unknown_surface_shows_empty_state(self, client: TestClient) -> None:
        """An unrecognised ?surface value must yield an empty-state page (not 500)."""
        r = client.get("/?surface=unknown")
        assert r.status_code == 200
        assert "no reports" in r.text.lower()


# ---------------------------------------------------------------------------
# Fix 3: read_json cache correctness
# ---------------------------------------------------------------------------


class TestFix3ParseCache:
    """Fix 3 — library.read_json is mtime-keyed and avoids redundant reads."""

    def test_cache_hit_on_second_call(self, tmp_path: Path) -> None:
        """Two calls with the same path + mtime must result in one read_text call."""
        # Clear any existing cache entries to get a clean slate.
        read_json.cache_clear()

        report_file = tmp_path / "rt_cache_test.json"
        report_file.write_text(json.dumps(_redteam_payload()))

        mtime_ns = report_file.stat().st_mtime_ns

        call_count = 0
        original_read_text = Path.read_text

        def _counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
            nonlocal call_count
            if self == report_file:
                call_count += 1
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(Path, "read_text", _counting_read_text):
            read_json(str(report_file.resolve()), mtime_ns)
            read_json(str(report_file.resolve()), mtime_ns)

        assert call_count == 1, f"Expected 1 read_text call, got {call_count}"

    def test_mtime_change_invalidates_cache(self, tmp_path: Path) -> None:
        """A new mtime must cause a fresh read instead of returning stale data."""
        read_json.cache_clear()

        report_file = tmp_path / "rt_mtime_test.json"
        payload_v1 = {**_redteam_payload(), "run_name": "v1"}
        report_file.write_text(json.dumps(payload_v1))
        mtime_v1 = report_file.stat().st_mtime_ns

        data_v1 = read_json(str(report_file.resolve()), mtime_v1)
        assert data_v1.get("run_name") == "v1"

        # Simulate a file update with a different mtime.
        payload_v2 = {**_redteam_payload(), "run_name": "v2"}
        report_file.write_text(json.dumps(payload_v2))
        mtime_v2 = report_file.stat().st_mtime_ns + 1  # guarantee different

        data_v2 = read_json(str(report_file.resolve()), mtime_v2)
        assert data_v2.get("run_name") == "v2"

    def test_cache_info_reports_hit(self, tmp_path: Path) -> None:
        """lru_cache.cache_info().hits must be > 0 after two identical calls."""
        read_json.cache_clear()

        report_file = tmp_path / "rt_info_test.json"
        report_file.write_text(json.dumps(_redteam_payload()))
        mtime_ns = report_file.stat().st_mtime_ns

        read_json(str(report_file.resolve()), mtime_ns)
        read_json(str(report_file.resolve()), mtime_ns)

        info = read_json.cache_info()
        assert info.hits >= 1


# ---------------------------------------------------------------------------
# Fix 4: filter apply() called once per POST
# ---------------------------------------------------------------------------


class TestFix4SingleFilterApply:
    """Fix 4 — recompute_options takes an already-filtered list (no double apply)."""

    def test_recompute_options_takes_filtered_list(self) -> None:
        """FILTERS['redteam'].recompute_options accepts a list, not (obj, sel)."""
        import inspect

        from evaluatorq.dashboard.filters import FILTERS

        sig = inspect.signature(FILTERS["redteam"].recompute_options)
        params = list(sig.parameters.keys())
        # New signature: a single positional param (the filtered list).
        assert len(params) == 1, f"Expected 1 param, got {params}"

    def test_sim_recompute_options_takes_filtered_list(self) -> None:
        """FILTERS['sim'].recompute_options accepts a list, not (obj, sel)."""
        import inspect

        from evaluatorq.dashboard.filters import FILTERS

        sig = inspect.signature(FILTERS["sim"].recompute_options)
        params = list(sig.parameters.keys())
        assert len(params) == 1, f"Expected 1 param, got {params}"

    def test_apply_called_once_during_filter_post(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """POST /r/{rid}/filter must call the underlying apply logic exactly once."""
        from evaluatorq.dashboard import filters as _filters_mod

        rt_path = roots[0] / "rt_fix_test.json"
        rid = report_id(rt_path)

        apply_calls: list[int] = []
        original_apply = _filters_mod._rt_apply

        def _counting_apply(report: object, selections: object) -> object:
            apply_calls.append(1)
            return original_apply(report, selections)  # type: ignore[arg-type]

        with patch.object(_filters_mod, "_rt_apply", _counting_apply):
            # Rebuild FILTERS with the patched apply so the route picks it up.
            from evaluatorq.dashboard.filters import FilterDef, _REDTEAM_DIMS, _rt_full_options, _rt_recompute_options

            patched_filter = FilterDef(
                dimensions=_REDTEAM_DIMS,
                options=_rt_full_options,
                apply=_counting_apply,
                recompute_options=_rt_recompute_options,  # type: ignore[arg-type]
            )
            original_filters = _filters_mod.FILTERS.copy()
            _filters_mod.FILTERS["redteam"] = patched_filter
            try:
                r = client.post(f"/r/{rid}/filter", data={"result": "All"})
            finally:
                _filters_mod.FILTERS["redteam"] = original_filters["redteam"]

        assert r.status_code == 200
        assert len(apply_calls) == 1, (
            f"apply() was called {len(apply_calls)} times; expected exactly 1"
        )

    def test_redteam_recompute_options_produces_correct_output(self) -> None:
        """_rt_recompute_options(filtered_list) must still return valid option dicts."""
        from evaluatorq.dashboard.filters import FILTERS

        from tests.redteam.reports.test_rebuild_filtered import _make_report, _make_result

        results = [
            _make_result(category="ASI01", passed=True),
            _make_result(category="ASI01", passed=False),
            _make_result(category="LLM01", passed=False),
        ]
        report = _make_report(results, tested_agents=["agent-a"])

        # Manually filter to ASI01 results.
        filtered = [r for r in report.results if r.attack.category == "ASI01"]

        opts = FILTERS["redteam"].recompute_options(filtered)
        assert "ASI01" in opts["category"]
        assert "LLM01" not in opts["category"]
        assert "result" in opts  # radio always present
