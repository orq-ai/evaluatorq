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
        assert '3 turns' in html
        assert 'score 0.82' in html
        assert 'Goal met' in html
        assert 'Goal missed' in html
        assert 'Error' in html

    def test_row_list_error_takes_precedence_over_goal_achieved(self) -> None:
        """terminated_by == 'error' tints error, regardless of goal_achieved."""
        from evaluatorq.dashboard.sim_views import render_sim_row_list

        entry = _entry(terminated_by='error', goal_achieved=True, error='boom')
        html = render_sim_row_list('rid', [entry])
        assert 'sim-tint-error' in html
        assert 'sim-tint-achieved' not in html


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
        # type label preserved as words (mockup shows "MUST NOT HAPPEN") + rendered red
        assert "must not happen" in html
        assert "sim-ctype-unsafe" in html
        assert "sim-judge" in html  # judge callout present when reason set

    def test_transcript_fragment_criteria_two_state_icons(self, sim_entry_with_safety_criterion) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_with_safety_criterion)
        assert 'sim-criterion-pass' in html
        assert 'sim-criterion-fail' in html

    def test_transcript_fragment_bubbles_present(self, sim_entry_with_transcript) -> None:
        from evaluatorq.dashboard.sim_views import render_transcript_fragment

        html = render_transcript_fragment(sim_entry_with_transcript)
        assert 'sim-msg-avatar' in html
        assert 'sim-transcript-grid' in html

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

        def _result(persona: str, marker: str) -> SimulationResult:
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
                metadata={'persona': persona, 'scenario': 's'},
            )

        results = [
            _result('alice', 'ALICE-ONE'),
            _result('alice', 'ALICE-TWO'),
            _result('bob', 'BOB-ONE'),
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
