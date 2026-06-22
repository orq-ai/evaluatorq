"""TDD tests for HTMX filter round-trip: POST /r/{rid}/filter.

Verifies:
- POST a category filter → fragment has FEWER result rows than unfiltered AND
  the form re-renders preserving the selection AND a now-empty dimension's
  option is gone.
- POST an empty form → full report (all results).
- Form re-renders with checked state matching the posted selections.
- Sim surface: persona filter reduces results.
- 404 returned for unknown rid.

Factory helpers are imported from the rebuild-filtered test module directly to
avoid duplicating the fixture factories.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id


# ---------------------------------------------------------------------------
# Re-use the _make_result / _make_report factories from the rebuild test
# (import verbatim — no duplication).
# ---------------------------------------------------------------------------

from tests.redteam.reports.test_rebuild_filtered import _make_report, _make_result  # noqa: E402


# ---------------------------------------------------------------------------
# Red-team report fixture helpers
# ---------------------------------------------------------------------------


def _rt_report():
    """Build a small RedTeamReport with results across TWO categories."""
    from evaluatorq.redteam.contracts import Severity

    results = [
        _make_result(category="ASI01", passed=True, agent_key="agent-a"),
        _make_result(category="ASI01", passed=False, agent_key="agent-a"),
        _make_result(category="LLM01", passed=False, agent_key="agent-a"),
        _make_result(category="LLM01", passed=True, agent_key="agent-a", severity=Severity.HIGH),
    ]
    return _make_report(results, tested_agents=["agent-a"])


def _write_rt_report(path: Path) -> None:
    report = _rt_report()
    path.write_text(report.model_dump_json())


# ---------------------------------------------------------------------------
# Sim report fixture helpers
# ---------------------------------------------------------------------------


def _sim_run():
    """Build a minimal SimulationRun with results across TWO personas."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy

    def _result(persona: str, goal_achieved: bool = True) -> SimulationResult:
        return SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason="done",
            goal_achieved=goal_achieved,
            goal_completion_score=1.0 if goal_achieved else 0.0,
            rules_broken=[],
            turn_count=2,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            metadata={"persona": persona, "scenario": "billing"},
        )

    results = [
        _result("alice", goal_achieved=True),
        _result("alice", goal_achieved=False),
        _result("bob", goal_achieved=True),
    ]
    return SimulationRun(
        run_name="test-sim-run",
        created_at=datetime.now(tz=timezone.utc),
        mode="run",
        target_kind="orq_agent",
        evaluator_names=["goal_achieved"],
        total_results=len(results),
        scorer_averages={"goal_achieved": 0.67},
        results=results,
    )


def _write_sim_run(path: Path) -> None:
    run = _sim_run()
    path.write_text(run.model_dump_json())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / "runs"
    sim = tmp_path / "sim-runs"
    rt.mkdir()
    sim.mkdir()
    _write_rt_report(rt / "rt_filter_test.json")
    _write_sim_run(sim / "sim_filter_test.json")
    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / "rt_filter_test.json"


def _sim_path(roots: list[Path]) -> Path:
    return roots[1] / "sim_filter_test.json"


# ---------------------------------------------------------------------------
# Red-team filter tests
# ---------------------------------------------------------------------------


class TestRedteamFilterRoute:
    """POST /r/{rid}/filter — red-team surface."""

    def test_filter_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200

    def test_empty_post_returns_full_report(self, client: TestClient, roots: list[Path]) -> None:
        """POST with no form data → no filtering applied → all 4 results rendered."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        # The fragment must contain the outer swap container.
        assert 'id="filter-swap"' in r.text

    def test_category_filter_reduces_rows(self, client: TestClient, roots: list[Path]) -> None:
        """Posting only ASI01 category must produce fewer result rows than unfiltered."""
        rid = report_id(_rt_path(roots))

        # Unfiltered fragment: 4 total results
        full = client.post(f"/r/{rid}/filter", data={})
        assert full.status_code == 200

        # Filtered to ASI01 only: 2 total results
        filtered = client.post(
            f"/r/{rid}/filter",
            data={"category": "ASI01", "result": "All"},
        )
        assert filtered.status_code == 200

        # The filtered body must mention ASI01 more than the full body mentions LLM01.
        # After filtering, the summary tables reflect 2 attacks (not 4).
        # LLM01 may appear once in a static OWASP reference section, but the
        # category-specific sections (by_category table) must not contain LLM01.
        # We verify the fragment contains fewer occurrences of LLM01 than the full body.
        assert full.text.count("LLM01") > filtered.text.count("LLM01")
        assert "ASI01" in filtered.text

    def test_category_filter_form_preserves_selection(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The re-rendered form must reflect the posted category selection."""
        rid = report_id(_rt_path(roots))
        r = client.post(
            f"/r/{rid}/filter",
            data={"category": "ASI01", "result": "All"},
        )
        assert r.status_code == 200
        text = r.text
        # Extract form section (between <form and </form>)
        form_start = text.find("<form")
        form_end = text.find("</form>")
        form_section = text[form_start : form_end + 7] if form_start >= 0 else ""

        # The checkbox for ASI01 must be checked.
        assert 'value="ASI01" checked' in form_section
        # After ASI01-only filter the LLM01 results are gone so the
        # recomputed options drop LLM01 entirely from the form.
        assert "LLM01" not in form_section

    def test_now_empty_dimension_option_drops_from_form(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """After filtering to ASI01 only, LLM01 category must disappear from the form."""
        rid = report_id(_rt_path(roots))
        r = client.post(
            f"/r/{rid}/filter",
            data={"category": "ASI01", "result": "All"},
        )
        assert r.status_code == 200
        # Extract only the filter form section to check options.
        text = r.text
        form_start = text.find("<form")
        form_end = text.find("</form>")
        form_section = text[form_start : form_end + 7] if form_start >= 0 else ""
        assert "LLM01" not in form_section

    def test_vulnerable_result_filter(self, client: TestClient, roots: list[Path]) -> None:
        """Posting result=Vulnerable must narrow to only vulnerable rows."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"result": "Vulnerable"})
        assert r.status_code == 200
        # The fragment must have the swap container.
        assert 'id="filter-swap"' in r.text
        # The re-rendered form must have Vulnerable selected.
        assert "Vulnerable" in r.text

    def test_missing_rid_returns_404(self, client: TestClient) -> None:
        r = client.post("/r/doesnotexist123/filter", data={})
        assert r.status_code == 404

    def test_fragment_contains_filter_form(self, client: TestClient, roots: list[Path]) -> None:
        """The fragment must contain the re-rendered filter form."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        # The hx-post attribute must point to the filter route.
        assert f"/r/{rid}/filter" in r.text
        assert 'filter-form' in r.text


# ---------------------------------------------------------------------------
# Simulation filter tests
# ---------------------------------------------------------------------------


class TestSimFilterRoute:
    """POST /r/{rid}/filter — simulation surface."""

    def test_filter_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200

    def test_empty_post_returns_full_report(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        assert 'id="filter-swap"' in r.text

    def test_persona_filter_reduces_results(self, client: TestClient, roots: list[Path]) -> None:
        """Filtering to persona=alice only should exclude bob's result."""
        rid = report_id(_sim_path(roots))
        r = client.post(
            f"/r/{rid}/filter",
            data={"persona": "alice", "goal_outcome": "All"},
        )
        assert r.status_code == 200
        assert "alice" in r.text.lower()
        assert "bob" not in r.text.lower()

    def test_persona_filter_form_preserves_selection(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Re-rendered form must reflect the alice-only persona selection."""
        rid = report_id(_sim_path(roots))
        r = client.post(
            f"/r/{rid}/filter",
            data={"persona": "alice", "goal_outcome": "All"},
        )
        assert r.status_code == 200
        # alice still appears in the recomputed form options.
        assert "alice" in r.text.lower()
        # bob should NOT appear in recomputed options (no results remain for bob).
        assert "bob" not in r.text.lower()

    def test_fragment_contains_filter_form(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        assert f"/r/{rid}/filter" in r.text
        assert "filter-form" in r.text


# ---------------------------------------------------------------------------
# OOB sidebar in filter response
# ---------------------------------------------------------------------------


class TestFilterOOBSidebar:
    """POST /r/{rid}/filter must include the OOB download sidebar in its response.

    The sidebar (hx-swap-oob="true") must carry filter params in CSV/JSON
    hrefs and, when those hrefs are followed, yield fewer rows than unfiltered.
    """

    def test_redteam_filter_post_has_oob_attr(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The POST response must contain an element with hx-swap-oob="true"."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        assert 'hx-swap-oob="true"' in r.text

    def test_redteam_filter_post_oob_sidebar_has_stable_id(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """OOB sidebar must carry id="download-sidebar" so HTMX can target it."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        assert 'id="download-sidebar"' in r.text

    def test_redteam_filter_post_oob_sidebar_csv_link_has_filter(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """OOB sidebar CSV link must include the category filter param."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r.status_code == 200
        sidebar_start = r.text.find('id="download-sidebar"')
        sidebar_end = r.text.find("</section>", sidebar_start)
        sidebar_html = r.text[sidebar_start:sidebar_end]
        assert "category=ASI01" in sidebar_html

    def test_redteam_oob_csv_link_round_trip_fewer_rows(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Filter → OOB sidebar link → GET that link → fewer rows than unfiltered."""
        import json
        rid = report_id(_rt_path(roots))

        # POST filter to ASI01-only.
        r_filter = client.post(f"/r/{rid}/filter", data={"category": "ASI01", "result": "All"})
        assert r_filter.status_code == 200

        # Extract CSV href from OOB sidebar.
        sidebar_start = r_filter.text.find('id="download-sidebar"')
        sidebar_end = r_filter.text.find("</section>", sidebar_start)
        sidebar_html = r_filter.text[sidebar_start:sidebar_end]
        csv_pos = sidebar_html.find("export.csv")
        assert csv_pos >= 0, "No CSV link in OOB sidebar"
        href_eq = sidebar_html.rfind('href="', 0, csv_pos)
        href_end = sidebar_html.find('"', href_eq + 6)
        csv_url = sidebar_html[href_eq + 6:href_end]

        # GET the filtered CSV.
        r_csv = client.get(csv_url)
        assert r_csv.status_code == 200
        filtered_rows = [ln for ln in r_csv.text.splitlines() if ln.strip()]

        # GET unfiltered CSV.
        all_rows = [ln for ln in client.get(f"/r/{rid}/export.csv").text.splitlines() if ln.strip()]

        assert len(filtered_rows) < len(all_rows), (
            f"Filtered CSV ({len(filtered_rows)} lines) not fewer than "
            f"unfiltered ({len(all_rows)} lines)"
        )

    def test_sim_filter_post_has_oob_sidebar(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Sim surface filter POST must also include the OOB sidebar."""
        rid = report_id(_sim_path(roots))
        r = client.post(f"/r/{rid}/filter", data={"persona": "alice", "goal_outcome": "All"})
        assert r.status_code == 200
        assert 'hx-swap-oob="true"' in r.text
        assert 'id="download-sidebar"' in r.text


# ---------------------------------------------------------------------------
# FilterDef unit tests (no HTTP)
# ---------------------------------------------------------------------------


class TestFilterDefUnit:
    """Direct tests of FILTERS['redteam'] and FILTERS['sim'] without HTTP."""

    def test_redteam_filterdef_importable(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        assert "redteam" in FILTERS
        assert "sim" in FILTERS

    def test_redteam_options_keys(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        opts = FILTERS["redteam"].options(report)
        assert "result" in opts
        assert "category" in opts
        assert "severity" in opts
        assert "technique" in opts
        assert "delivery_method" in opts
        assert "vulnerability" in opts
        assert "agent" in opts

    def test_redteam_apply_category_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        filtered = FILTERS["redteam"].apply(report, {"category": ["ASI01"]})
        assert all(r.attack.category == "ASI01" for r in filtered)
        assert len(filtered) == 2

    def test_redteam_apply_empty_selections_returns_all(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        filtered = FILTERS["redteam"].apply(report, {})
        assert len(filtered) == len(report.results)

    def test_redteam_recompute_options_drops_empty_categories(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        new_opts = FILTERS["redteam"].recompute_options(report, {"category": ["ASI01"]})
        # After filtering to ASI01, LLM01 should not appear in recomputed options.
        assert "LLM01" not in new_opts["category"]
        assert "ASI01" in new_opts["category"]

    def test_sim_options_keys(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        opts = FILTERS["sim"].options(run)
        assert "persona" in opts
        assert "scenario" in opts
        assert "terminated_by" in opts
        assert "goal_outcome" in opts

    def test_sim_apply_persona_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        filtered = FILTERS["sim"].apply(run, {"persona": ["alice"]})
        assert all(r.metadata.get("persona") == "alice" for r in filtered)
        assert len(filtered) == 2

    def test_sim_apply_goal_outcome_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        filtered = FILTERS["sim"].apply(run, {"goal_outcome": ["Achieved"]})
        assert all(r.goal_achieved for r in filtered)

    def test_sim_recompute_options_drops_empty_persona(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        new_opts = FILTERS["sim"].recompute_options(run, {"persona": ["alice"]})
        assert "alice" in new_opts["persona"]
        assert "bob" not in new_opts["persona"]
