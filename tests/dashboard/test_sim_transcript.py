"""TDD tests for the sim transcript drill-down route.

Verifies:
- GET /r/{rid}/sim/transcript?idx=0   → 200, HTML, persona/scenario in output
- GET /r/{rid}/sim/transcript?idx=N   → graceful empty (no 500) for out-of-range idx
- Bad idx param (non-integer)         → graceful (falls back to idx=0)
- Missing rid                         → 404
- Transcript messages rendered in markup (role + content)
- XSS: a message containing <script> appears escaped
- Sim row list is embedded in the report page (section element present)
- Redteam report → 404 on transcript route (surface mismatch)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.contracts import TokenUsage
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_sim_run_with_transcript() -> SimulationRun:
    """Build a SimulationRun with real transcript messages for drill-down testing."""
    from evaluatorq.contracts import Message

    results = [
        SimulationResult(
            messages=[
                Message(role="user", content="Hello, I need help with my order."),
                Message(role="assistant", content="Sure, what is your order number?"),
                Message(role="user", content="Order 12345."),
            ],
            terminated_by=TerminatedBy.judge,
            reason="Goal achieved after 3 turns",
            goal_achieved=True,
            goal_completion_score=0.95,
            rules_broken=[],
            turn_count=3,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=20, output_tokens=15, total_tokens=35),
            metadata={"persona": "alice", "scenario": "billing inquiry"},
        ),
        SimulationResult(
            messages=[
                Message(role="user", content="I want a refund!"),
                Message(role="assistant", content="I understand. Let me check."),
            ],
            terminated_by=TerminatedBy.max_turns,
            reason="Max turns reached",
            goal_achieved=False,
            goal_completion_score=0.2,
            rules_broken=[],
            turn_count=2,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=8, total_tokens=18),
            metadata={"persona": "bob", "scenario": "refund request"},
        ),
    ]
    return SimulationRun(
        run_name="transcript-test-run",
        created_at=datetime.now(tz=timezone.utc),
        mode="run",
        target_kind="orq_agent",
        evaluator_names=["goal_achieved"],
        total_results=len(results),
        scorer_averages={"goal_achieved": 0.5},
        results=results,
    )


def _make_xss_sim_run() -> SimulationRun:
    """Build a SimulationRun where a message contains a raw XSS payload."""
    from evaluatorq.contracts import Message

    results = [
        SimulationResult(
            messages=[
                Message(
                    role="user",
                    content="Hello",
                ),
                Message(
                    role="assistant",
                    # Malicious agent message — stored-XSS vector.
                    content='<script>alert("xss")</script>',
                ),
            ],
            terminated_by=TerminatedBy.judge,
            reason="done",
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=1,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
            metadata={"persona": "attacker", "scenario": "injection test"},
        )
    ]
    return SimulationRun(
        run_name="xss-test-run",
        created_at=datetime.now(tz=timezone.utc),
        mode="run",
        target_kind="orq_agent",
        evaluator_names=["goal_achieved"],
        total_results=1,
        scorer_averages={"goal_achieved": 1.0},
        results=results,
    )


def _make_rt_report():
    """Minimal redteam report for the surface-mismatch 404 test."""
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

    results = [
        RedTeamResult(
            attack=AttackInfo(
                id="rt-1",
                category="ASI01",
                vulnerability="",
                framework=Framework.OWASP_ASI,
                attack_technique=AttackTechnique.INDIRECT_INJECTION,
                delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
                turn_type=TurnType.SINGLE,
                severity=Severity.MEDIUM,
                source="test",
            ),
            agent=AgentInfo(key="agent-a"),
            messages=[],
            vulnerable=False,
            response="ok",
            evaluation=UnifiedEvaluationResult(passed=True, explanation="ok"),
        )
    ]
    summary = compute_report_summary(results)
    return RedTeamReport(
        pipeline=Pipeline.STATIC,
        created_at=datetime.now(tz=timezone.utc),
        categories_tested=["ASI01"],
        total_results=len(results),
        results=results,
        summary=summary,
        description="rt-for-transcript-test",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    sim_dir = tmp_path / "sim-runs"
    rt_dir = tmp_path / "runs"
    sim_dir.mkdir()
    rt_dir.mkdir()

    sim_run = _make_sim_run_with_transcript()
    (sim_dir / "sim_transcript_test.json").write_text(sim_run.model_dump_json())

    rt_report = _make_rt_report()
    (rt_dir / "rt_transcript_test.json").write_text(rt_report.model_dump_json())

    return [rt_dir, sim_dir]


@pytest.fixture()
def xss_roots(tmp_path: Path) -> list[Path]:
    sim_dir = tmp_path / "sim-runs"
    sim_dir.mkdir()
    xss_run = _make_xss_sim_run()
    (sim_dir / "xss_sim.json").write_text(xss_run.model_dump_json())
    return [sim_dir]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def xss_client(xss_roots: list[Path]) -> TestClient:
    app = build_app(roots=xss_roots)
    return TestClient(app, raise_server_exceptions=True)


def _sim_path(roots: list[Path]) -> Path:
    return roots[1] / "sim_transcript_test.json"


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / "rt_transcript_test.json"


def _xss_path(xss_roots: list[Path]) -> Path:
    return xss_roots[0] / "xss_sim.json"


# ---------------------------------------------------------------------------
# Transcript route: basic behaviour
# ---------------------------------------------------------------------------


class TestSimTranscriptRoute:
    def test_transcript_idx0_returns_200(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert r.status_code == 200

    def test_transcript_content_type_is_html(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert "text/html" in r.headers.get("content-type", "")

    def test_transcript_contains_persona(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert "alice" in r.text

    def test_transcript_contains_scenario(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert "billing" in r.text.lower()

    def test_transcript_contains_message_markup(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        # The transcript messages must appear wrapped in sim-msg markup.
        assert "sim-msg" in r.text

    def test_transcript_contains_first_message_content(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert "Hello, I need help with my order" in r.text

    def test_transcript_contains_metrics(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        # Metrics section must be present (turns, score, goal)
        assert "sim-transcript-metrics" in r.text or "sim-metric" in r.text

    def test_transcript_contains_judge_reason(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        # judge reason "Goal achieved after 3 turns" should appear
        assert "Goal achieved" in r.text

    def test_transcript_idx1_returns_second_conversation(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=1")
        assert r.status_code == 200
        assert "bob" in r.text

    def test_transcript_out_of_range_idx_no_500(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=999")
        # Must not 500 — graceful empty or 200 with empty message.
        assert r.status_code != 500

    def test_transcript_non_integer_idx_no_500(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=abc")
        assert r.status_code != 500

    def test_transcript_missing_rid_returns_404(
        self, client: TestClient
    ) -> None:
        r = client.get("/r/nonexistent123/sim/transcript?idx=0")
        assert r.status_code == 404

    def test_transcript_redteam_rid_returns_404(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Transcript route must return 404 when rid is a redteam report."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# XSS escaping
# ---------------------------------------------------------------------------


class TestSimTranscriptXssEscaping:
    def test_script_tag_is_escaped(
        self, xss_client: TestClient, xss_roots: list[Path]
    ) -> None:
        rid = report_id(_xss_path(xss_roots))
        r = xss_client.get(f"/r/{rid}/sim/transcript?idx=0")
        assert r.status_code == 200
        # The raw <script> tag must NOT appear verbatim in the response.
        assert "<script>" not in r.text
        # The escaped form must appear instead.
        assert "&lt;script&gt;" in r.text


# ---------------------------------------------------------------------------
# Sim row list embedded in the report page
# ---------------------------------------------------------------------------


class TestSimRowListOnReportPage:
    def test_sim_report_page_contains_row_list(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert "sim-row-list" in r.text or "sim-row-table" in r.text

    def test_sim_report_page_contains_transcript_panel(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "sim-transcript-panel" in r.text

    def test_sim_report_page_has_hx_get_links(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        # Each row must have an hx-get pointing to the transcript endpoint.
        assert "/sim/transcript" in r.text

    def test_sim_report_page_shows_persona_names(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "alice" in r.text
        assert "bob" in r.text


# ---------------------------------------------------------------------------
# Filter-awareness: sim transcript and row-list honor filter query params
# ---------------------------------------------------------------------------


class TestSimFilterAwareness:
    """Verify that the sim transcript route applies filter params from the query-string.

    The filter dimensions (persona, scenario, terminated_by, goal_outcome) are
    carried by hx-include="#filter-form" so each transcript hx-get request
    automatically includes the active filter selections.
    """

    def test_transcript_with_persona_filter_returns_200(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Transcript route with a persona filter param must return 200."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0&persona=alice")
        assert r.status_code == 200

    def test_transcript_persona_filter_shows_matching_entry(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """When filtering to persona=alice, idx=0 in the filtered list maps to alice."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0&persona=alice")
        assert r.status_code == 200
        assert "alice" in r.text

    def test_transcript_filter_drops_non_matching_persona(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """When filtering to persona=alice, bob should not appear in idx=0."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=0&persona=alice")
        assert r.status_code == 200
        # With only alice in the filtered set, idx=0 is alice's entry.
        assert "bob" not in r.text

    def test_transcript_out_of_range_after_filter_is_graceful(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Filtering to a single persona leaves 1 entry; idx=1 must not 500."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/transcript?idx=1&persona=alice")
        # Out-of-range idx after filtering → graceful empty, not 500.
        assert r.status_code in (200, 404)
        assert r.status_code != 500

    def test_sim_row_list_route_returns_200(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /r/{rid}/sim/row-list must return 200 with row-list HTML."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/row-list")
        assert r.status_code == 200
        assert "sim-row" in r.text or "sim-transcript-panel" in r.text

    def test_sim_row_list_with_filter_returns_fewer_rows(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """Filtered row-list should contain fewer persona rows than unfiltered."""
        rid = report_id(_sim_path(roots))
        html_all = client.get(f"/r/{rid}/sim/row-list").text
        html_alice = client.get(f"/r/{rid}/sim/row-list?persona=alice").text
        # Count sim-row-item occurrences as proxy for number of rows.
        all_count = html_all.count("sim-row-item")
        alice_count = html_alice.count("sim-row-item")
        assert alice_count < all_count, (
            f"Expected fewer rows when filtering to persona=alice: "
            f"unfiltered={all_count}, filtered={alice_count}"
        )

    def test_sim_row_list_missing_rid_returns_404(
        self, client: TestClient
    ) -> None:
        """row-list route for an unknown rid must return 404."""
        r = client.get("/r/nonexistent-sim/sim/row-list")
        assert r.status_code == 404

    def test_sim_panel_container_has_hx_include(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The sim interactive panel container must include hx-include='#filter-form'."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert 'hx-include="#filter-form"' in r.text, (
            "Expected hx-include=\"#filter-form\" on sim panel container so "
            "filter selections are carried into transcript hx-get requests."
        )

    def test_sim_panel_container_triggers_on_filter_changed(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The sim panel container must listen for orq:filter-changed."""
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert "orq:filter-changed" in r.text, (
            "Expected 'orq:filter-changed' in hx-trigger on sim panel container."
        )

    def test_filter_post_emits_hx_trigger_for_sim(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """POST /r/{rid}/filter for a sim report must return HX-Trigger header."""
        rid = report_id(_sim_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        hx_trigger = r.headers.get("hx-trigger", "")
        assert "orq:filter-changed" in hx_trigger, (
            f"Expected HX-Trigger: orq:filter-changed, got: {hx_trigger!r}"
        )
