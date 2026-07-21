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
                Message(role='user', content='Hello, I need help with my order.'),
                Message(role='assistant', content='Sure, what is your order number?'),
                Message(role='user', content='Order 12345.'),
            ],
            terminated_by=TerminatedBy.judge,
            reason='Goal achieved after 3 turns',
            goal_achieved=True,
            goal_completion_score=0.95,
            rules_broken=[],
            turn_count=3,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=20, output_tokens=15, total_tokens=35),
            metadata={'persona': 'alice', 'scenario': 'billing inquiry'},
        ),
        SimulationResult(
            messages=[
                Message(role='user', content='I want a refund!'),
                Message(role='assistant', content='I understand. Let me check.'),
            ],
            terminated_by=TerminatedBy.max_turns,
            reason='Max turns reached',
            goal_achieved=False,
            goal_completion_score=0.2,
            rules_broken=[],
            turn_count=2,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=8, total_tokens=18),
            metadata={'persona': 'bob', 'scenario': 'refund request'},
        ),
    ]
    return SimulationRun(
        run_name='transcript-test-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=len(results),
        scorer_averages={'goal_achieved': 0.5},
        results=results,
    )


def _make_xss_sim_run() -> SimulationRun:
    """Build a SimulationRun where a message contains a raw XSS payload."""
    from evaluatorq.contracts import Message

    results = [
        SimulationResult(
            messages=[
                Message(
                    role='user',
                    content='Hello',
                ),
                Message(
                    role='assistant',
                    # Malicious agent message — stored-XSS vector.
                    content='<script>alert("xss")</script>',
                ),
            ],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=1,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
            metadata={'persona': 'attacker', 'scenario': 'injection test'},
        )
    ]
    return SimulationRun(
        run_name='xss-test-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=1,
        scorer_averages={'goal_achieved': 1.0},
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
                id='rt-1',
                category='ASI01',
                vulnerability='',
                framework=Framework.OWASP_ASI,
                attack_technique=AttackTechnique.INDIRECT_INJECTION,
                delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
                turn_type=TurnType.SINGLE,
                severity=Severity.MEDIUM,
                source='test',
            ),
            agent=AgentInfo(key='agent-a'),
            messages=[],
            vulnerable=False,
            response='ok',
            evaluation=UnifiedEvaluationResult(passed=True, explanation='ok'),
        )
    ]
    summary = compute_report_summary(results)
    return RedTeamReport(
        pipeline=Pipeline.STATIC,
        created_at=datetime.now(tz=timezone.utc),
        categories_tested=['ASI01'],
        total_results=len(results),
        results=results,
        summary=summary,
        description='rt-for-transcript-test',
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    sim_dir = tmp_path / 'sim-runs'
    rt_dir = tmp_path / 'runs'
    sim_dir.mkdir()
    rt_dir.mkdir()

    sim_run = _make_sim_run_with_transcript()
    (sim_dir / 'sim_transcript_test.json').write_text(sim_run.model_dump_json())

    rt_report = _make_rt_report()
    (rt_dir / 'rt_transcript_test.json').write_text(rt_report.model_dump_json())

    return [rt_dir, sim_dir]


@pytest.fixture()
def xss_roots(tmp_path: Path) -> list[Path]:
    sim_dir = tmp_path / 'sim-runs'
    sim_dir.mkdir()
    xss_run = _make_xss_sim_run()
    (sim_dir / 'xss_sim.json').write_text(xss_run.model_dump_json())
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
    return roots[1] / 'sim_transcript_test.json'


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / 'rt_transcript_test.json'


def _xss_path(xss_roots: list[Path]) -> Path:
    return xss_roots[0] / 'xss_sim.json'


# ---------------------------------------------------------------------------
# Transcript route: basic behaviour
# ---------------------------------------------------------------------------


class TestSimTranscriptRoute:
    def test_transcript_idx0_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        assert r.status_code == 200

    def test_transcript_content_type_is_html(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        assert 'text/html' in r.headers.get('content-type', '')

    def test_transcript_contains_persona(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        # Persona/scenario now live on the collapsed conversation card (row-list),
        # not in the transcript fragment — the fragment no longer repeats them.
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/row-list")
        assert "alice" in r.text

    def test_transcript_contains_scenario(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}/sim/row-list")
        assert "billing" in r.text.lower()

    def test_transcript_contains_message_markup(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # The transcript messages must appear wrapped in sim-msg markup.
        assert 'sim-msg' in r.text

    def test_transcript_contains_first_message_content(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        assert 'Hello, I need help with my order' in r.text

    def test_transcript_contains_grid_structure(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # Design-aligned fragment: bubbles | criteria grid, no separate metrics block.
        assert 'sim-transcript-grid' in r.text
        assert 'sim-criteria' in r.text

    def test_transcript_summary_shows_persona_scenario_and_turns(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # Conversation summary header recaps persona + scenario and a turn chip.
        assert 'sim-conv-summary' in r.text
        assert 'sim-conv-turns-pill' in r.text
        assert '3 turns' in r.text  # fixture idx0 has turn_count=3
        assert 'sim-conv-index' in r.text  # teal #index badge, top-left

    def test_transcript_persona_scenario_are_clickthrough(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # Persona/scenario values are cohort-card triggers, ids matching the
        # `persona-{i}` / `scenario-{i}` templates rendered on the report page.
        assert 'data-sim-entity-trigger data-entity-kind="persona" data-entity-id="persona-0"' in r.text
        assert 'data-sim-entity-trigger data-entity-kind="scenario" data-entity-id="scenario-0"' in r.text
        assert 'sim-conv-value--link' in r.text

    def test_transcript_judge_folded_into_criteria(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # Judge rationale lives inside the criteria block, not a standalone callout.
        assert 'sim-judge' in r.text
        crit = r.text.index('sim-criteria')
        assert crit < r.text.index('sim-judge') < r.text.index('sim-transcript-bubbles')

    def test_transcript_criteria_precedes_conversation(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # Criteria block must render before the chat bubbles inside the grid.
        assert r.text.index('sim-criteria') < r.text.index('sim-transcript-bubbles')

    def test_transcript_contains_judge_reason(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        # judge reason "Goal achieved after 3 turns" should appear
        assert 'Goal achieved' in r.text

    def test_transcript_idx1_returns_second_conversation(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=1')
        assert r.status_code == 200
        # Second conversation is bob's — identify it by his unique refund content
        # (the fragment no longer embeds the persona label).
        assert "refund" in r.text.lower()

    def test_transcript_out_of_range_idx_no_500(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=999')
        # Must not 500 — graceful empty or 200 with empty message.
        assert r.status_code != 500

    def test_transcript_non_integer_idx_no_500(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=abc')
        assert r.status_code != 500

    def test_transcript_missing_rid_returns_404(self, client: TestClient) -> None:
        r = client.get('/r/nonexistent123/sim/transcript?idx=0')
        assert r.status_code == 404

    def test_transcript_redteam_rid_returns_404(self, client: TestClient, roots: list[Path]) -> None:
        """Transcript route must return 404 when rid is a redteam report."""
        rid = report_id(_rt_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0')
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# XSS escaping
# ---------------------------------------------------------------------------


class TestSimTranscriptXssEscaping:
    def test_script_tag_is_escaped(self, xss_client: TestClient, xss_roots: list[Path]) -> None:
        rid = report_id(_xss_path(xss_roots))
        r = xss_client.get(f'/r/{rid}/sim/transcript?idx=0')
        assert r.status_code == 200
        # The raw <script> tag must NOT appear verbatim in the response.
        assert '<script>' not in r.text
        # The escaped form must appear instead.
        assert '&lt;script&gt;' in r.text


# ---------------------------------------------------------------------------
# Sim row list embedded in the report page
# ---------------------------------------------------------------------------


class TestSimRowListOnReportPage:
    def test_sim_report_page_contains_row_list(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        assert 'sim-row-list' in r.text or 'sim-row-table' in r.text

    def test_sim_report_page_has_clickable_conversation_rows(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f"/r/{rid}")
        assert "sim-conv-row" in r.text
        assert 'data-entity-kind="conversation"' in r.text

    def test_sim_report_page_has_transcript_drawer_urls(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}')
        # Each row must carry a lazy drawer URL for the transcript endpoint.
        assert '/sim/transcript' in r.text

    def test_sim_report_page_shows_persona_names(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}')
        assert 'alice' in r.text
        assert 'bob' in r.text


# ---------------------------------------------------------------------------
# Filter-awareness: sim transcript and row-list honor filter query params
# ---------------------------------------------------------------------------


class TestSimFilterAwareness:
    """Verify row-list filtering while transcript lookup remains full-run indexed."""

    def test_transcript_with_persona_filter_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        """Transcript route with a persona filter param must return 200."""
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0&persona=alice')
        assert r.status_code == 200

    def test_transcript_persona_filter_shows_matching_entry(self, client: TestClient, roots: list[Path]) -> None:
        """Full-run idx=0 remains Alice when an irrelevant filter is supplied."""
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0&persona=alice')
        assert r.status_code == 200
        # Identify the conversation by its (unique) message content — the fragment
        # no longer embeds the persona label (that lives on the card).
        assert "Order 12345" in r.text

    def test_transcript_filter_drops_non_matching_persona(self, client: TestClient, roots: list[Path]) -> None:
        """When filtering to persona=alice, bob should not appear in idx=0."""
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=0&persona=alice')
        assert r.status_code == 200
        # Full-run idx=0 is Alice, so Bob's unique refund content cannot appear.
        assert "refund" not in r.text.lower()

    def test_transcript_out_of_range_after_filter_is_graceful(self, client: TestClient, roots: list[Path]) -> None:
        """An optional filter parameter does not make transcript lookup raise."""
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/transcript?idx=1&persona=alice')
        # Full-run lookup remains graceful regardless of the filter parameter.
        assert r.status_code in (200, 404)
        assert r.status_code != 500

    def test_sim_row_list_route_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        """GET /r/{rid}/sim/row-list must return 200 with row-list HTML."""
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}/sim/row-list')
        assert r.status_code == 200
        assert 'sim-row' in r.text or 'sim-conv-row' in r.text

    def test_sim_row_list_with_filter_returns_fewer_rows(self, client: TestClient, roots: list[Path]) -> None:
        """Filtered row-list should contain fewer persona rows than unfiltered."""
        rid = report_id(_sim_path(roots))
        html_all = client.get(f'/r/{rid}/sim/row-list').text
        html_alice = client.get(f'/r/{rid}/sim/row-list?persona=alice').text
        # Count sim-row-item occurrences as proxy for number of rows.
        all_count = html_all.count('sim-conv-row')
        alice_count = html_alice.count('sim-conv-row')
        assert alice_count < all_count, (
            f'Expected fewer rows when filtering to persona=alice: unfiltered={all_count}, filtered={alice_count}'
        )

    def test_sim_row_list_missing_rid_returns_404(self, client: TestClient) -> None:
        """row-list route for an unknown rid must return 404."""
        r = client.get('/r/nonexistent-sim/sim/row-list')
        assert r.status_code == 404

    def test_sim_rowlist_wrapper_no_longer_self_refetches(self, client: TestClient, roots: list[Path]) -> None:
        """Double-fetch removal: the row-list wrapper div itself carries no
        hx-include/hx-trigger of its own — the /filter POST body swap is the
        single refresh path (spec §Transcripts double-fetch fix).

        Conversation drawer rows no longer issue their own HTMX requests; this
        test only guards the outer wrapper div, which must stay an inert
        container with no hx-trigger of its own.
        """
        rid = report_id(_sim_path(roots))
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        wrapper_start = r.text.find('id="sim-row-list-')
        assert wrapper_start >= 0
        wrapper_tag_end = r.text.find('>', wrapper_start)
        wrapper_tag = r.text[max(0, wrapper_start - 20) : wrapper_tag_end + 1]
        assert 'hx-include' not in wrapper_tag
        assert 'hx-trigger' not in wrapper_tag
        assert 'orq:filter-changed' not in r.text

    def test_filter_post_emits_hx_trigger_for_sim(self, client: TestClient, roots: list[Path]) -> None:
        """POST /r/{rid}/filter for a sim report must return HX-Trigger header."""
        rid = report_id(_sim_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200
        hx_trigger = r.headers.get('hx-trigger', '')
        assert 'orq:filter-changed' in hx_trigger, f'Expected HX-Trigger: orq:filter-changed, got: {hx_trigger!r}'


# ---------------------------------------------------------------------------
# Direct-function tests: conversation cards (Task 11) + transcript fragment
# rewrite (Task 12) — design-aligned <details> cards, judge callout, bubbles,
# two-state criteria (deviation #4: the old three-state ⛔ safety icon is
# removed; safety is preserved via a red "must_not_happen" type label).
# ---------------------------------------------------------------------------


def _entry(
    *,
    index: int = 0,
    persona: str = 'alice',
    scenario: str = 'billing inquiry',
    terminated_by: str = 'judge',
    goal_achieved: bool = True,
    goal_completion_score: float = 0.82,
    turn_count: int = 3,
    judge_reason: str = 'Goal achieved after 3 turns.',
    error: str | None = None,
    criteria: list | None = None,
    transcript: list | None = None,
):
    from evaluatorq.simulation.types import CriteriaRow, SimulationEntry, TranscriptMessage

    if criteria is None:
        criteria = [
            CriteriaRow(
                id='c1',
                description='Agent confirms the order number',
                type='must_happen',
                passed=True,
                safety=False,
            ),
        ]
    if transcript is None:
        transcript = [
            TranscriptMessage(role='user', content='hi'),
            TranscriptMessage(role='assistant', content='yo'),
        ]
    return SimulationEntry(
        index=index,
        persona=persona,
        scenario=scenario,
        model='gpt-4o',
        target_model='gpt-4o',
        terminated_by=terminated_by,
        goal_achieved=goal_achieved,
        goal_completion_score=goal_completion_score,
        rules_broken=[],
        criteria=criteria,
        turn_count=turn_count,
        total_tokens=100,
        judge_reason=judge_reason,
        error=error,
        evaluator_scores={},
        transcript=transcript,
    )


@pytest.fixture()
def sim_entries():
    return [
        _entry(index=0, persona='alice', goal_achieved=True, terminated_by='judge'),
        _entry(index=1, persona='bob', goal_achieved=False, terminated_by='max_turns'),
        _entry(index=2, persona='carol', terminated_by='error', error='boom'),
    ]


@pytest.fixture()
def sim_entry_with_safety_criterion():
    from evaluatorq.simulation.types import CriteriaRow

    return _entry(
        criteria=[
            CriteriaRow(
                id='c1',
                description='Agent must not reveal internal credentials',
                type='must_not_happen',
                passed=True,
                safety=True,
            ),
            CriteriaRow(
                id='c2',
                description='Agent confirms order number',
                type='must_happen',
                passed=False,
                safety=False,
            ),
        ],
    )


@pytest.fixture()
def sim_entry_with_transcript():
    return _entry()


@pytest.fixture()
def sim_entry_xss_criterion():
    from evaluatorq.simulation.types import CriteriaRow

    return _entry(
        criteria=[
            CriteriaRow(
                id='c1',
                description='<script>alert("xss")</script>',
                type='must_happen',
                passed=True,
                safety=False,
            ),
        ],
    )


class TestConversationRows:
    """Conversation list entries are lazy drawer triggers, not foldout cards."""

    def test_row_list_renders_clickable_conversation_rows(self, sim_entries) -> None:
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        html = render_sim_row_list('rid', sim_entries)
        assert '<details' not in html
        assert 'sim-conv-card' not in html
        assert html.count('data-sim-entity-trigger') == len(sim_entries)
        assert 'data-entity-kind="conversation"' in html
        assert 'data-drawer-url="/r/rid/sim/transcript?idx=0"' in html
        assert 'role="button" tabindex="0"' in html
        assert 'hx-trigger="toggle once' not in html

    def test_dashboard_runtime_has_no_failure_anchor_handler(self) -> None:
        source = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()

        assert 'a[href^="#conv-"]' not in source

    def test_row_list_summary_has_header_cluster(self, sim_entries) -> None:
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        html = render_sim_row_list('rid', sim_entries)
        assert '#1' in html
        assert 'alice' in html
        assert '<td class="sim-conv-turns">3</td>' in html
        assert '<td class="sim-conv-score">0.82</td>' in html
        assert 'Goal met' in html
        assert 'Goal missed' in html
        assert 'Error' in html

    def test_trace_anchor_carries_no_drawer_optout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The trace link lives in a row cell (aligned columns), so it nests
        inside the ``role=button`` row. ``data-no-drawer`` is what keeps a
        trace-link click/keypress from also firing the drawer (dashboard.js
        bails on the ``[data-no-drawer]`` ancestor for both events)."""
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        monkeypatch.setenv('ORQ_WORKSPACE_SLUG', 'workspace')
        entry = _entry().model_copy(update={'thread_id': 'run:0'})
        html = render_sim_row_list('rid', [entry])

        row_start = html.index('<tr class="sim-conv-row ')
        row_end = html.index('</tr>', row_start)
        row = html[row_start:row_end]

        assert 'role="button" tabindex="0"' in row
        assert 'data-sim-entity-trigger' in row
        # The trace anchor is inside the row but opts out of the drawer.
        trace_start = row.index('<a class="btn-secondary trace-link"')
        assert 'data-no-drawer' in row[trace_start:]

    def test_row_list_error_takes_precedence_over_goal_achieved(self) -> None:
        """terminated_by == 'error' tints error, regardless of goal_achieved."""
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        entry = _entry(terminated_by='error', goal_achieved=True, error='boom')
        html = render_sim_row_list('rid', [entry])
        assert 'sim-tint-error' in html
        assert 'sim-tint-achieved' not in html

    def test_row_list_sorts_by_column(self) -> None:
        """Sorting reorders visible rows without renumbering their index links."""
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        entries = [
            _entry(index=0, persona='carol', turn_count=5),
            _entry(index=1, persona='alice', turn_count=9),
            _entry(index=2, persona='bob', turn_count=1),
        ]
        html = render_sim_row_list('rid', entries, sort='turn_count', direction='desc')
        positions = [html.index(f'>{p}<') for p in ('alice', 'carol', 'bob')]  # 9, 5, 1
        assert positions == sorted(positions)
        assert 'aria-sort="descending"' in html
        # Index links stay stable — sorting reorders display, not the run position.
        assert 'transcript?idx=1' in html

    def test_row_list_paginates(self) -> None:
        """Only page_size rows render per page; the pager reflects the split."""
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        entries = [_entry(index=i, persona=f'p{i}') for i in range(30)]
        page1 = render_sim_row_list('rid', entries, page=1, page_size=25)
        page2 = render_sim_row_list('rid', entries, page=2, page_size=25)
        assert page1.count('sim-conv-row') == 25
        assert page2.count('sim-conv-row') == 5
        assert 'Page 1 of 2' in page1
        # Out-of-range page clamps to the last page rather than rendering empty.
        assert 'Page 2 of 2' in render_sim_row_list('rid', entries, page=99, page_size=25)

    def test_pager_hidden_for_single_page(self) -> None:
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        html = render_sim_row_list('rid', [_entry(index=0)])
        assert 'sim-pager-btn' not in html
        assert '1 conversations' in html
        # Size selector shows even with a single page so you can shrink from 25.
        assert 'sim-size' in html

    def test_row_list_size_selector(self) -> None:
        """Size selector offers 5/10/25; the active size is the disabled/current one
        and other options re-fetch with that size and page reset to 1."""
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        entries = [_entry(index=i, persona=f'p{i}') for i in range(30)]
        html = render_sim_row_list('rid', entries, page=1, page_size=10)
        assert html.count('sim-conv-row') == 10
        assert 'Page 1 of 3' in html
        # Active option (10) is disabled + aria-current; others link with size=.
        assert 'aria-current="true">10<' in html
        assert 'size=5&' not in html  # size is the last query param, no trailing &
        assert 'page=1&size=5"' in html
        assert 'page=1&size=25"' in html

    def test_page_size_coercion_rejects_bad_values(self) -> None:
        from evaluatorq.dashboard.sim_views import _PAGE_SIZE, _coerce_page_size

        assert _coerce_page_size('5') == 5
        assert _coerce_page_size('10') == 10
        assert _coerce_page_size('25') == 25
        assert _coerce_page_size('7') == _PAGE_SIZE  # not an allowed option
        assert _coerce_page_size(None) == _PAGE_SIZE
        assert _coerce_page_size('abc') == _PAGE_SIZE


class TestRowlistWrapperNoSelfRefetch:
    def test_rowlist_wrapper_no_self_refetch(self) -> None:
        from evaluatorq.dashboard.view import _sim_rowlist_wrapper

        html = _sim_rowlist_wrapper('rid', '<section></section>')
        assert 'orq:filter-changed' not in html
        assert 'hx-include' not in html


class TestMessageListAvatarAndSide:
    """Task 12: render_message_list avatar/side extension."""

    def test_message_list_has_avatar_and_side(self) -> None:
        from evaluatorq.dashboard.view import render_message_list

        msgs = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'yo'}]
        html = render_message_list(msgs, role_labels={'user': 'USR', 'assistant': 'AGT'}, class_prefix='sim')
        assert 'sim-msg-avatar' in html
        assert 'sim-msg-user' in html and 'sim-msg-assistant' in html
        assert 'USR' in html and 'AGT' in html


class TestTranscriptFragmentRewrite:
    """Task 12: judge callout + bubbles + two-state criteria."""

    def test_transcript_fragment_two_state_criteria(self, sim_entry_with_safety_criterion) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_with_safety_criterion)
        assert "⛔" not in html  # three-state ⛔ icon gone
        assert "&#x26D4;" not in html
        # Polarity chip sits beside the result icon; must_not_happen reads red.
        assert "Prohibited" in html
        assert "sim-ctype-unsafe" in html
        assert "sim-judge" in html  # judge callout present when reason set

    def test_transcript_fragment_criteria_two_state_icons(self, sim_entry_with_safety_criterion) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_with_safety_criterion)
        assert 'sim-criterion-pass' in html
        assert 'sim-criterion-fail' in html

    def test_transcript_fragment_criteria_verdict(self, sim_entry_with_safety_criterion) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_with_safety_criterion)
        # goal_achieved defaults True, one of two criteria passed. Goal outcome and
        # criteria tally are shown as separate spans (not "PASS · 1/2 met").
        assert 'sim-criteria-verdict--pass' in html
        assert 'Goal met' in html
        assert '1/2 criteria met' in html

    def test_transcript_fragment_bubbles_present(self, sim_entry_with_transcript) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_with_transcript)
        assert 'sim-msg-avatar' in html
        assert 'sim-transcript-grid' in html

    def test_drawer_stacks_criteria_above_conversation_and_swaps_message_sides(self) -> None:
        """The wide drawer stacks criteria above the transcript, divided by a rule."""
        from evaluatorq.dashboard.styles import DASHBOARD_CSS

        assert '.sim-report .sim-entity-dialog' in DASHBOARD_CSS
        assert 'width: 60vw;' in DASHBOARD_CSS
        assert '.sim-report .sim-transcript-grid { display: flex; flex-direction: column; gap: 0; }' in DASHBOARD_CSS
        # Hairline divider between stacked sections (criteria ↔ conversation).
        assert '.sim-report .sim-transcript-grid > * + * {' in DASHBOARD_CSS
        # User right / agent left. Must reset BOTH margins + flex-direction so the
        # shared `.report-aligned` base rules (which also match the drawer) can't
        # leave the opposite margin at `auto` and center the bubble.
        # Margin shorthand carries the side-swap (auto) AND the vertical gap in the
        # bottom slot — a `margin: 0 X 0 auto` form would zero out margin-bottom.
        assert '.sim-report .sim-msg-user, .sim-report .sim-msg-system { margin: 0 0 16px auto; flex-direction: row-reverse; }' in DASHBOARD_CSS
        assert '.sim-report .sim-msg-assistant, .sim-report .sim-msg-tool { margin: 0 auto 16px 0; flex-direction: row; }' in DASHBOARD_CSS

    def test_backdrop_close_waits_for_drawer_exit_animation(self) -> None:
        source = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()
        from evaluatorq.dashboard.styles import DASHBOARD_CSS

        assert 'function closeDrawer()' in source
        assert "dialog.addEventListener('animationend', finishClose, { once: true });" in source
        assert 'sim-entity-dialog--closing' in source
        assert '.sim-report .sim-entity-dialog--closing { animation: sim-drawer-out 160ms ease-in forwards; }' in DASHBOARD_CSS

    def test_drawer_drill_pushes_browser_history(self) -> None:
        """Each persona/scenario/conversation drill is a real history entry so the
        browser Back/Forward buttons walk the drill path."""
        source = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()

        assert "history.pushState({ simDrawer: serial, drawerDepth: drawerDepth }, '')" in source
        assert "window.addEventListener('popstate'" in source
        assert 'evt.state.simDrawer' in source
        # Back button and native Escape unwind through history, not a private stack.
        assert 'history.back()' in source
        assert 'history.go(-drawerDepth)' in source

    def test_transcript_fragment_error_entry_shows_error_message(self) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        entry = _entry(terminated_by='error', error='the target crashed')
        html = render_transcript_fragment(entry)
        assert 'the target crashed' in html
        assert 'sim-transcript-error' in html


class TestTranscriptCriteriaXssEscaping:
    def test_transcript_criteria_escapes_html(self, sim_entry_xss_criterion) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_xss_criterion)
        assert '<script>' not in html
        assert '&lt;script&gt;' in html


# ---------------------------------------------------------------------------
# Regression: conversation drawer URLs keep their full-run identity after a
# filter response. ``sim_transcript`` deliberately resolves ``idx`` against
# the full run, so the filtered row must retain that original index.
# ---------------------------------------------------------------------------


class TestFilteredConversationDrawerIndex:
    def test_filtered_row_idx_resolves_to_same_persona_via_transcript_route(self, tmp_path: Path) -> None:
        """End-to-end guard: with personas [alice, alice, bob] and a persona=bob
        filter active, bob's card carries his STABLE idx (2 — his position in
        the full run), not a re-numbered filtered idx. The transcript route
        resolves idx against the full, unfiltered run, so the drill-down returns
        bob's conversation regardless of the active filter — and an unfiltered
        row idx can never overflow a shorter filtered list (the original bug).
        """
        from evaluatorq.contracts import Message, TokenUsage
        from evaluatorq.dashboard.app import build_app
        from evaluatorq.simulation.types import (
            SimulationResult,
            SimulationRun,
            TerminatedBy,
        )

        def _result(persona: str, scenario: str, marker: str) -> SimulationResult:
            return SimulationResult(
                messages=[
                    Message(role='user', content=f'hi from {marker}'),
                    Message(role='assistant', content=f'reply to {marker}'),
                ],
                terminated_by=TerminatedBy.judge,
                reason='done',
                goal_achieved=True,
                goal_completion_score=1.0,
                rules_broken=[],
                turn_count=1,
                turn_metrics=[],
                token_usage=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
                metadata={'persona': persona, 'scenario': scenario},
            )

        results = [
            _result('alice', 'account access', 'ALICE-ONE'),
            _result('alice', 'account access', 'ALICE-TWO'),
            _result('bob', 'billing', 'BOB-ONE'),
        ]
        run = SimulationRun(
            run_name='filtered-idx-test',
            created_at=datetime.now(tz=timezone.utc),
            mode='run',
            target_kind='orq_agent',
            evaluator_names=['goal_achieved'],
            total_results=len(results),
            scorer_averages={'goal_achieved': 1.0},
            results=results,
        )
        sim_dir = tmp_path / 'sim-runs'
        sim_dir.mkdir()
        run_path = sim_dir / 'filtered_idx_test.json'
        run_path.write_text(run.model_dump_json())

        app = build_app(roots=[sim_dir])
        client = TestClient(app, raise_server_exceptions=True)
        rid = report_id(run_path)

        # Filter the report to persona=bob — this is the full /filter response
        # that replaces the report body in the browser. Bob is the only visible
        # entry, but his drawer row must carry
        # his *stable* idx (2 — his position in the full [alice, alice, bob]
        # run), because filtering hides rows rather than renumbering them.
        filtered_html = client.post(f'/r/{rid}/filter', data={'persona': 'bob'}).text
        drawer_url = f'/r/{rid}/sim/transcript?idx=2'
        assert drawer_url in filtered_html

        # Config and its drawer-template registry stay full-run, even though
        # Breakdown remains filtered. The configured but excluded Alice/account
        # cohorts are still actionable and explicitly show an empty cohort.
        import re

        persona_match = re.search(
            r'<button[^>]*class="sim-config-persona-row sim-entity-row"[^>]*'
            r'data-entity-kind="persona" data-entity-id="([^"]+)"[^>]*>'
            r'.*?<span class="sim-config-persona-name">alice</span>',
            filtered_html,
        )
        scenario_match = re.search(
            r'<button[^>]*class="sim-config-scenario-row sim-entity-row"[^>]*'
            r'data-entity-kind="scenario" data-entity-id="([^"]+)"[^>]*>'
            r'.*?<span class="sim-config-scenario-name">account access</span>',
            filtered_html,
        )
        assert persona_match is not None and scenario_match is not None
        for entity_id in (persona_match.group(1), scenario_match.group(1)):
            template = re.search(rf'<template id="{re.escape(entity_id)}"[^>]*>(.*?)</template>', filtered_html)
            assert template is not None
            assert '<p class="sim-cohort-empty">No conversations.</p>' in template.group(1)
        assert '<td data-label="Persona">alice</td>' not in filtered_html
        assert '<td data-label="Scenario">account access</td>' not in filtered_html

        # The endpoint's full-run resolver maps Bob's stable idx directly to
        # Bob; no filter parameters are needed for identity resolution.
        r = client.get(drawer_url)
        assert r.status_code == 200
        assert 'BOB-ONE' in r.text, (
            f"Expected bob's conversation from the filtered drawer URL "
            f'({drawer_url!r}), got: '
            f'{r.text!r}'
        )
        assert 'ALICE-ONE' not in r.text
        assert 'ALICE-TWO' not in r.text
