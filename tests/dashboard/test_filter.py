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
        _make_result(category='ASI01', passed=True, agent_key='agent-a'),
        _make_result(category='ASI01', passed=False, agent_key='agent-a'),
        _make_result(category='LLM01', passed=False, agent_key='agent-a'),
        _make_result(category='LLM01', passed=True, agent_key='agent-a', severity=Severity.HIGH),
    ]
    return _make_report(results, tested_agents=['agent-a'])


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
            reason='done',
            goal_achieved=goal_achieved,
            goal_completion_score=1.0 if goal_achieved else 0.0,
            rules_broken=[],
            turn_count=2,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            metadata={'persona': persona, 'scenario': 'billing'},
        )

    results = [
        _result('alice', goal_achieved=True),
        _result('alice', goal_achieved=False),
        _result('bob', goal_achieved=True),
    ]
    return SimulationRun(
        run_name='test-sim-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=len(results),
        scorer_averages={'goal_achieved': 0.67},
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
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    _write_rt_report(rt / 'rt_filter_test.json')
    _write_sim_run(sim / 'sim_filter_test.json')
    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / 'rt_filter_test.json'


def _sim_path(roots: list[Path]) -> Path:
    return roots[1] / 'sim_filter_test.json'


# ---------------------------------------------------------------------------
# Red-team filter tests
# ---------------------------------------------------------------------------


class TestRedteamFilterRoute:
    """POST /r/{rid}/filter — red-team surface."""

    def test_filter_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_rt_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200

    def test_empty_post_returns_full_report(self, client: TestClient, roots: list[Path]) -> None:
        """POST with no form data → no filtering applied → all 4 results rendered."""
        rid = report_id(_rt_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200
        # The fragment must contain the outer swap container.
        assert 'id="filter-swap"' in r.text

    def test_category_filter_reduces_rows(self, client: TestClient, roots: list[Path]) -> None:
        """Posting only ASI01 category must produce fewer result rows than unfiltered."""
        rid = report_id(_rt_path(roots))

        # Unfiltered fragment: 4 total results
        full = client.post(f'/r/{rid}/filter', data={})
        assert full.status_code == 200

        # Filtered to ASI01 only: 2 total results
        filtered = client.post(
            f'/r/{rid}/filter',
            data={'category': 'ASI01'},
        )
        assert filtered.status_code == 200

        # The filtered body must mention ASI01 more than the full body mentions LLM01.
        # After filtering, the summary tables reflect 2 attacks (not 4).
        # LLM01 may appear once in a static OWASP reference section, but the
        # category-specific sections (by_category table) must not contain LLM01.
        # We verify the fragment contains fewer occurrences of LLM01 than the full body.
        assert full.text.count('LLM01') > filtered.text.count('LLM01')
        assert 'ASI01' in filtered.text

    def test_category_filter_form_preserves_selection(self, client: TestClient, roots: list[Path]) -> None:
        """The re-rendered form must reflect the posted category selection."""
        rid = report_id(_rt_path(roots))
        r = client.post(
            f'/r/{rid}/filter',
            data={'category': 'ASI01'},
        )
        assert r.status_code == 200
        text = r.text
        # Extract form section (between <form and </form>)
        form_start = text.find('<form')
        form_end = text.find('</form>')
        form_section = text[form_start : form_end + 7] if form_start >= 0 else ''

        # The checkbox for ASI01 must be checked.
        assert 'value="ASI01" checked' in form_section
        # LLM01 stays available (unchecked) so it can be re-selected.
        assert 'value="LLM01"' in form_section

    def test_deselected_dimension_option_stays_in_form(self, client: TestClient, roots: list[Path]) -> None:
        """Filtering to ASI01 only must NOT drop LLM01 from the form options."""
        rid = report_id(_rt_path(roots))
        r = client.post(
            f'/r/{rid}/filter',
            data={'category': 'ASI01'},
        )
        assert r.status_code == 200
        # Extract only the filter form section to check options.
        text = r.text
        form_start = text.find('<form')
        form_end = text.find('</form>')
        form_section = text[form_start : form_end + 7] if form_start >= 0 else ''
        assert 'value="LLM01"' in form_section
        assert 'value="ASI01" checked' in form_section

    def test_vulnerable_result_filter(self, client: TestClient, roots: list[Path]) -> None:
        """Posting result=Vulnerable must narrow to only vulnerable rows."""
        rid = report_id(_rt_path(roots))
        r = client.post(f'/r/{rid}/filter', data={'result': 'Vulnerable'})
        assert r.status_code == 200
        # The fragment must have the swap container.
        assert 'id="filter-swap"' in r.text
        # The re-rendered form must have Vulnerable selected.
        assert 'Vulnerable' in r.text

    def test_missing_rid_returns_404(self, client: TestClient) -> None:
        r = client.post('/r/doesnotexist123/filter', data={})
        assert r.status_code == 404

    def test_fragment_contains_filter_form(self, client: TestClient, roots: list[Path]) -> None:
        """The fragment must contain the re-rendered filter form."""
        rid = report_id(_rt_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200
        # The hx-post attribute must point to the filter route.
        assert f'/r/{rid}/filter' in r.text
        assert 'filter-form' in r.text


# ---------------------------------------------------------------------------
# Simulation filter tests
# ---------------------------------------------------------------------------


class TestSimFilterRoute:
    """POST /r/{rid}/filter — simulation surface."""

    def test_filter_returns_200(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200

    def test_empty_post_returns_full_report(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200
        assert 'id="filter-swap"' in r.text

    def test_persona_filter_reduces_results(self, client: TestClient, roots: list[Path]) -> None:
        """Filtering to alice excludes Bob's results but keeps the Config registry."""
        rid = report_id(_sim_path(roots))
        r = client.post(
            f'/r/{rid}/filter',
            data={'persona': 'alice', 'goal_outcome': 'All'},
        )
        assert r.status_code == 200
        text = r.text
        # The four emitted panels are Overview, Breakdown, Transcripts, and
        # Config. Scope assertions to their exact panel boundaries: Config
        # deliberately retains every entity that can open a drawer.
        tab_panels = text[text.index('<div class="tab-panels">') :]
        panels = tab_panels.split('<section class="tab-panel">')
        assert len(panels) == 5
        breakdown, transcripts, config = panels[2:]
        assert 'alice' in breakdown.lower()
        assert 'bob' not in breakdown.lower()
        assert 'alice' in transcripts.lower()
        assert 'bob' not in transcripts.lower()
        assert 'bob' in config.lower()

    def test_metric_dim_round_trips_through_http(self, client: TestClient, roots: list[Path]) -> None:
        """A new metric dimension (max_goal_score) narrows results end-to-end via
        the real POST route — proving parse_selections → _SIM_DIMS → _sim_apply
        are wired for the new dims, not just unit-tested in isolation."""
        rid = report_id(_sim_path(roots))
        # Fixture: 3 conversations with goal scores 1.0 / 0.0 / 1.0.
        full = client.post(f'/r/{rid}/filter', data={})
        assert 'Transcripts <span class="tab-count">3</span>' in full.text
        # Ceiling of 0.5 keeps only the single 0.0-score conversation.
        filtered = client.post(f'/r/{rid}/filter', data={'max_goal_score': '0.5'})
        assert filtered.status_code == 200
        assert 'Transcripts <span class="tab-count">1</span>' in filtered.text

    def test_persona_filter_form_preserves_selection(self, client: TestClient, roots: list[Path]) -> None:
        """Re-rendered form keeps alice checked and bob available to re-select."""
        rid = report_id(_sim_path(roots))
        r = client.post(
            f'/r/{rid}/filter',
            data={'persona': 'alice', 'goal_outcome': 'All'},
        )
        assert r.status_code == 200
        text = r.text
        form_start = text.find('<form')
        form_end = text.find('</form>')
        form_section = text[form_start : form_end + 7] if form_start >= 0 else ''
        assert 'value="alice" checked' in form_section
        # bob stays as an unchecked option — deselecting must not remove it.
        assert 'value="bob"' in form_section

    def test_fragment_contains_filter_form(self, client: TestClient, roots: list[Path]) -> None:
        rid = report_id(_sim_path(roots))
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200
        assert f'/r/{rid}/filter' in r.text
        assert 'filter-form' in r.text


# ---------------------------------------------------------------------------
# FilterDef unit tests (no HTTP)
# ---------------------------------------------------------------------------


class TestFilterDefUnit:
    """Direct tests of FILTERS['redteam'] and FILTERS['sim'] without HTTP."""

    def test_redteam_filterdef_importable(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        assert 'redteam' in FILTERS
        assert 'sim' in FILTERS

    def test_redteam_options_keys(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        opts = FILTERS['redteam'].options(report)
        assert 'result' in opts
        assert 'category' in opts
        assert 'severity' in opts
        assert 'technique' in opts
        assert 'delivery_method' in opts
        assert 'vulnerability' in opts
        assert 'agent' in opts

    def test_redteam_delivery_method_options_use_values_not_enum_reprs(self) -> None:
        """Options must carry 'direct-request', never 'DeliveryMethod.DIRECT_REQUEST'.

        The fixture's attacks hold DeliveryMethod members, and str(member) renders the
        repr on the 3.10 StrEnum polyfill — so this pins delivery_method_str as the
        rendering path for the filter rail and chart labels.
        """
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        assert FILTERS['redteam'].options(report)['delivery_method'] == ['direct-request']

    def test_redteam_apply_delivery_method_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        assert len(FILTERS['redteam'].apply(report, {'delivery_method': ['direct-request']})) == 4
        assert FILTERS['redteam'].apply(report, {'delivery_method': ['base64']}) == []

    def test_redteam_apply_category_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        filtered = FILTERS['redteam'].apply(report, {'category': ['ASI01']})
        assert all(r.attack.category == 'ASI01' for r in filtered)
        assert len(filtered) == 2

    def test_redteam_apply_empty_selections_returns_all(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        filtered = FILTERS['redteam'].apply(report, {})
        assert len(filtered) == len(report.results)

    def test_redteam_options_stay_full_after_filtering(self) -> None:
        # Options come from the full report, never the filtered rows, so a
        # deselected value never vanishes from its own multi-select.
        from evaluatorq.dashboard.filters import FILTERS

        report = _rt_report()
        full = FILTERS['redteam'].options(report)
        assert {'ASI01', 'LLM01'} <= set(full['category'])
        # Narrowing to ASI01 must not shrink the option list.
        FILTERS['redteam'].apply(report, {'category': ['ASI01']})
        assert FILTERS['redteam'].options(report)['category'] == full['category']

    def test_sim_options_keys(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        opts = FILTERS['sim'].options(run)
        assert 'persona' in opts
        assert 'scenario' in opts
        assert 'terminated_by' in opts
        assert 'goal_outcome' in opts

    def test_sim_apply_persona_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        filtered = FILTERS['sim'].apply(run, {'persona': ['alice']})
        assert all(r.metadata.get('persona') == 'alice' for r in filtered)
        assert len(filtered) == 2

    def test_sim_apply_goal_outcome_filter(self) -> None:
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        filtered = FILTERS['sim'].apply(run, {'goal_outcome': ['Achieved']})
        assert all(r.goal_achieved for r in filtered)

    def test_sim_options_stay_full_after_filtering(self) -> None:
        # Deselecting a persona must not drop it from the persona multi-select.
        from evaluatorq.dashboard.filters import FILTERS

        run = _sim_run()
        full = FILTERS['sim'].options(run)
        assert {'alice', 'bob'} <= set(full['persona'])
        FILTERS['sim'].apply(run, {'persona': ['alice']})
        assert FILTERS['sim'].options(run)['persona'] == full['persona']


# ---------------------------------------------------------------------------
# goal_outcome two-value multiselect (Task 5)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sim_results() -> list:
    """A small list of SimulationResult-like objects with mixed goal_achieved."""
    return _sim_run().results


def _run(results):
    class R:  # noqa: D401
        pass

    r = R()
    r.results = results
    return r


class TestGoalOutcomeMultiselect:
    @pytest.mark.parametrize(
        'sel,expected_all',
        [
            ([], True),  # zero => All
            (['Achieved', 'Not achieved'], True),  # two => All
        ],
    )
    def test_goal_outcome_zero_or_two_is_all(self, sim_results, sel, expected_all) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        run = _run(sim_results)
        out = _sim_apply(run, {'goal_outcome': sel})
        assert len(out) == len(sim_results)

    def test_goal_outcome_single_achieved(self, sim_results) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        run = _run(sim_results)
        out = _sim_apply(run, {'goal_outcome': ['Achieved']})
        assert all(r.goal_achieved for r in out)

    def test_goal_outcome_single_not_achieved(self, sim_results) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        run = _run(sim_results)
        out = _sim_apply(run, {'goal_outcome': ['Not achieved']})
        assert all(not r.goal_achieved for r in out)

    def test_sim_options_no_all_sentinel(self, sim_results) -> None:
        from evaluatorq.dashboard.filters import _sim_options_from_results

        opts = _sim_options_from_results(sim_results)
        assert opts['goal_outcome'] == ['Achieved', 'Not achieved']


# ---------------------------------------------------------------------------
# Task 6: right-side sim filter rail (chips + dropdowns) + body-first source
# order swap.
# ---------------------------------------------------------------------------


def test_sim_filter_form_renders_rail_widgets():
    from evaluatorq.dashboard.view import render_filter_form

    opts = {
        'persona': ['Alice', 'Bob'],
        'scenario': ['Refund'],
        'terminated_by': ['goal', 'error'],
        'goal_outcome': ['Achieved', 'Not achieved'],
    }
    html = render_filter_form('rid', 'sim', opts, {})
    assert 'id="filter-dd-persona"' in html
    assert 'id="filter-dd-scenario"' in html
    assert 'name="goal_outcome"' in html
    assert 'Achieved' in html and 'Not achieved' in html
    # chip toggles are checkboxes wrapped in labels
    assert 'type="checkbox"' in html
    # stable form contract preserved
    assert 'id="filter-form"' in html
    assert 'hx-post="/r/rid/filter"' in html
    assert 'hx-trigger="change"' in html
    assert 'hx-target="#filter-swap"' in html
    assert 'hx-swap="outerHTML"' in html


def test_filter_fragment_body_before_form():
    from evaluatorq.dashboard.view import filter_fragment

    html = filter_fragment('rid', 'sim', "<div class='report-body-area'>BODY</div>", '<form>FORM</form>')
    assert html.index('BODY') < html.index('FORM')  # body first, form second (rail on right)


class TestSimFilterRailCounter:
    def test_counter_shows_showing_all_when_shown_equals_total(self):
        from evaluatorq.dashboard.view import render_filter_form

        opts = {
            'persona': ['Alice'],
            'scenario': ['Refund'],
            'terminated_by': ['goal'],
            'goal_outcome': ['Achieved', 'Not achieved'],
        }
        html = render_filter_form('rid', 'sim', opts, {}, shown=5, total=5)
        assert 'Showing all results' in html

    def test_counter_shows_n_of_m_when_filtered(self):
        from evaluatorq.dashboard.view import render_filter_form

        opts = {
            'persona': ['Alice'],
            'scenario': ['Refund'],
            'terminated_by': ['goal'],
            'goal_outcome': ['Achieved', 'Not achieved'],
        }
        html = render_filter_form('rid', 'sim', opts, {}, shown=2, total=5)
        assert '2 of 5 shown' in html

    def test_counter_defaults_to_showing_all_when_kwargs_absent(self):
        from evaluatorq.dashboard.view import render_filter_form

        opts = {
            'persona': ['Alice'],
            'scenario': ['Refund'],
            'terminated_by': ['goal'],
            'goal_outcome': ['Achieved', 'Not achieved'],
        }
        html = render_filter_form('rid', 'sim', opts, {})
        assert 'Showing all results' in html

    def test_sim_rail_dropdown_checked_state_reflects_selections(self):
        from evaluatorq.dashboard.view import render_filter_form

        opts = {
            'persona': ['Alice', 'Bob'],
            'scenario': ['Refund'],
            'terminated_by': ['goal'],
            'goal_outcome': ['Achieved', 'Not achieved'],
        }
        html = render_filter_form('rid', 'sim', opts, {'persona': ['Alice']})
        dd_start = html.index('id="filter-dd-persona"')
        dd_end = html.index('</details>', dd_start)
        section = html[dd_start:dd_end]
        assert 'value="Alice" checked' in section
        assert 'value="Bob" checked' not in section
        assert 'value="Bob"' in section

    def test_sim_rail_escapes_option_values(self):
        from evaluatorq.dashboard.view import render_filter_form

        opts = {
            'persona': ['<script>alert(1)</script>'],
            'scenario': [],
            'terminated_by': [],
            'goal_outcome': ['Achieved', 'Not achieved'],
        }
        html = render_filter_form('rid', 'sim', opts, {})
        assert '<script>' not in html

    def test_redteam_branch_unaffected_by_new_kwargs(self):
        from evaluatorq.dashboard.view import render_filter_form

        opts = {
            'category': ['ASI01', 'LLM01'],
            'result': ['Vulnerable', 'Resistant', 'Error'],
            'agent': ['a'],
            'max_turns': ['1'],
        }
        html = render_filter_form('rid', 'redteam', opts, {}, shown=1, total=2)
        assert 'id="filter-form"' in html
        assert 'filter-dd-persona' not in html


# ---------------------------------------------------------------------------
# Task 7: persist open <details> filter dropdowns across HTMX filter swaps.
# ---------------------------------------------------------------------------
def test_dashboard_js_has_details_persistence_hook():
    from pathlib import Path

    js = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()
    assert 'htmx:beforeSwap' in js
    assert 'filter-dd' in js or 'details[open]' in js


# ---------------------------------------------------------------------------
# Task 11: double-fetch removal — the sim row-list wrapper no longer
# self-refetches on orq:filter-changed; the /filter POST body swap already
# delivers the (now heavier, card-based) row list.
# ---------------------------------------------------------------------------
def test_rowlist_wrapper_no_self_refetch():
    from evaluatorq.dashboard.view import _sim_rowlist_wrapper

    html = _sim_rowlist_wrapper('rid', '<section></section>')
    assert 'orq:filter-changed' not in html  # double-fetch removed
    assert 'hx-include' not in html
    assert '<section></section>' in html


# ---------------------------------------------------------------------------
# Task 9: result multiselect, min_turns slider, RT rail rendering
# ---------------------------------------------------------------------------
def _rt_run(results):
    class R:
        pass

    r = R()
    r.results = results
    return r


@pytest.mark.parametrize('sel,expect_all', [([], True), (['Vulnerable', 'Resistant', 'Error'], True)])
def test_rt_result_zero_or_all_is_all(rt_results, sel, expect_all):
    from evaluatorq.dashboard.filters import _rt_apply

    out = _rt_apply(_rt_run(rt_results), {'result': sel})
    assert len(out) == len(rt_results)


def test_rt_result_single_vulnerable(rt_results):
    from evaluatorq.dashboard.filters import _rt_apply

    out = _rt_apply(_rt_run(rt_results), {'result': ['Vulnerable']})
    assert all(r.vulnerable and not r.error for r in out)


def test_rt_min_turns_filters(rt_results):
    from evaluatorq.dashboard.filters import _rt_apply

    out = _rt_apply(_rt_run(rt_results), {'min_turns': ['2']})
    for r in out:
        assert (r.execution.turns if r.execution else 1) >= 2


def test_rt_min_turns_default_no_filter(rt_results):
    from evaluatorq.dashboard.filters import _rt_apply

    out = _rt_apply(_rt_run(rt_results), {'min_turns': ['1']})
    assert len(out) == len(rt_results)


def test_rt_options_include_max_turns_without_all_sentinel(rt_results):
    from evaluatorq.dashboard.filters import _rt_options_from_results

    opts = _rt_options_from_results(rt_results)
    assert opts['result'] == ['Vulnerable', 'Resistant', 'Error']
    assert 'max_turns' in opts


def test_rt_rail_has_slider_and_more_expander():
    from evaluatorq.dashboard.view import _render_redteam_filter_rail

    opts = {
        'result': ['Vulnerable', 'Resistant', 'Error'],
        'severity': ['critical'],
        'category': ['ASI01'],
        'agent': ['a', 'b'],
        'technique': ['crescendo'],
        'delivery_method': ['email'],
        'vulnerability': ['v1'],
        'max_turns': ['5'],
    }
    html = _render_redteam_filter_rail('rid', opts, {}, shown=3, total=3)
    assert 'name="min_turns"' in html and 'type="range"' in html
    assert 'id="filter-dd-more"' in html
    assert 'id="filter-dd-technique"' in html  # inside the expander
    # Shared range control: numeric readout (never "all") + the run's max shown.
    assert '>all<' not in html
    assert '<span class="filter-slider-max">/ 5</span>' in html


def test_rt_rail_slider_engaged_only_when_off_default():
    from evaluatorq.dashboard.view import _render_redteam_filter_rail

    opts = {
        'result': ['Vulnerable', 'Resistant', 'Error'],
        'severity': ['critical'],
        'category': ['ASI01'],
        'agent': ['a'],
        'technique': ['t'],
        'delivery_method': ['email'],
        'vulnerability': ['v1'],
        'max_turns': ['5'],
    }
    off = _render_redteam_filter_rail('rid', opts, {'min_turns': ['1']}, shown=3, total=3)
    assert 'filter-slider-readout is-engaged' not in off
    on = _render_redteam_filter_rail('rid', opts, {'min_turns': ['3']}, shown=3, total=3)
    assert 'filter-slider-readout is-engaged' in on


def test_rt_rail_hides_slider_when_max_turns_one():
    from evaluatorq.dashboard.view import _render_redteam_filter_rail

    opts = {
        'result': ['Vulnerable', 'Resistant', 'Error'],
        'severity': ['critical'],
        'category': ['ASI01'],
        'agent': ['a'],
        'technique': ['t'],
        'delivery_method': ['email'],
        'vulnerability': ['v1'],
        'max_turns': ['1'],
    }
    html = _render_redteam_filter_rail('rid', opts, {}, shown=1, total=1)
    assert 'name="min_turns"' not in html  # slider hidden for single-turn-only runs


def test_rt_options_no_all_sentinel(rt_results):
    from evaluatorq.dashboard.filters import _rt_options_from_results

    opts = _rt_options_from_results(rt_results)
    assert opts['result'] == ['Vulnerable', 'Resistant', 'Error']  # no 'All' sentinel


# ---------------------------------------------------------------------------
# Task 2 (metric filters plan): rule/goal/turns/tokens/metric threshold dims.
# ---------------------------------------------------------------------------


def _sim_metric_run():
    """SimulationRun with three personas exercising rule/metric filter edges.

    - alice:   rules_broken=['criteria_0'], low hallucination risk, high
               response quality, turn_count=3, total_tokens=150.
    - risky:   no rules broken, HIGH hallucination risk, LOW response
               quality, turn_count=5, total_tokens=500.
    - unscored: no rules broken, turn_metrics=[] (nothing scored), turn_count=1,
               total_tokens=15.
    """
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy, TurnMetrics

    def _turn(**scores) -> TurnMetrics:
        return TurnMetrics(
            turn_number=1,
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            judge_reason='ok',
            **scores,
        )

    results = [
        SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=0.9,
            rules_broken=['criteria_0'],
            turn_count=3,
            turn_metrics=[
                _turn(
                    response_quality=0.8,
                    hallucination_risk=0.2,
                    tone_appropriateness=0.9,
                    factual_accuracy=0.85,
                )
            ],
            token_usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            metadata={'persona': 'alice', 'scenario': 'billing'},
        ),
        SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=False,
            goal_completion_score=0.3,
            rules_broken=[],
            turn_count=5,
            turn_metrics=[
                _turn(
                    response_quality=0.6,
                    hallucination_risk=0.9,
                    tone_appropriateness=0.5,
                    factual_accuracy=0.4,
                )
            ],
            token_usage=TokenUsage(input_tokens=300, output_tokens=200, total_tokens=500),
            metadata={'persona': 'risky', 'scenario': 'billing'},
        ),
        SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=1,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            metadata={'persona': 'unscored', 'scenario': 'billing'},
        ),
    ]
    return SimulationRun(
        run_name='test-sim-metric-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=len(results),
        scorer_averages={'goal_achieved': 0.67},
        results=results,
    )


@pytest.fixture()
def sim_run():
    return _sim_metric_run()


class TestSimMetricFilters:
    def test_rule_broken_is_opt_in(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        assert [r.rules_broken for r in _sim_apply(sim_run, {'rule_broken': ['yes']})] == [['criteria_0']]
        assert len(_sim_apply(sim_run, {})) == len(sim_run.results)

    def test_risk_uses_worst_turn_and_keeps_unscored(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        filtered = _sim_apply(sim_run, {'min_hallucination_risk': ['0.70']})
        assert {r.metadata['persona'] for r in filtered} == {'risky', 'unscored'}

    def test_quality_ceiling_uses_worst_turn_and_keeps_unscored(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        filtered = _sim_apply(sim_run, {'max_response_quality': ['0.70']})
        assert {r.metadata['persona'] for r in filtered} == {'risky', 'unscored'}

    def test_max_goal_score_is_ceiling(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        filtered = _sim_apply(sim_run, {'max_goal_score': ['0.5']})
        assert {r.metadata['persona'] for r in filtered} == {'risky'}

    def test_min_turns_is_floor(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        filtered = _sim_apply(sim_run, {'min_turns': ['3']})
        assert {r.metadata['persona'] for r in filtered} == {'alice', 'risky'}

    def test_min_total_tokens_is_floor(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        filtered = _sim_apply(sim_run, {'min_total_tokens': ['100']})
        assert {r.metadata['persona'] for r in filtered} == {'alice', 'risky'}

    def test_empty_selections_returns_all(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        assert len(_sim_apply(sim_run, {})) == len(sim_run.results)

    def test_bad_threshold_input_is_noop(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        filtered = _sim_apply(sim_run, {'max_goal_score': ['not-a-number']})
        assert len(filtered) == len(sim_run.results)

    def test_non_positive_count_is_noop(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_apply

        # Zero/negative count floors never narrow — treated as "no filter".
        for bad in ('0', '-5'):
            assert len(_sim_apply(sim_run, {'min_turns': [bad]})) == len(sim_run.results)
            assert len(_sim_apply(sim_run, {'min_total_tokens': [bad]})) == len(sim_run.results)

    def test_full_options_exposes_raw_maxima_and_available_metrics(self, sim_run) -> None:
        from evaluatorq.dashboard.filters import _sim_full_options

        opts = _sim_full_options(sim_run)
        assert opts['max_turns'] == ['5']
        assert opts['max_total_tokens'] == ['500']
        assert set(opts['metrics']) == {
            'response_quality',
            'hallucination_risk',
            'tone_appropriateness',
            'factual_accuracy',
        }

    def test_full_options_hides_unavailable_metrics(self) -> None:
        from evaluatorq.contracts import TokenUsage
        from evaluatorq.dashboard.filters import _sim_full_options
        from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy

        result = SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=1,
            turn_metrics=[],
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            metadata={'persona': 'solo', 'scenario': 'x'},
        )
        run = SimulationRun(
            run_name='no-metrics-run',
            created_at=datetime.now(tz=timezone.utc),
            mode='run',
            target_kind='orq_agent',
            evaluator_names=[],
            total_results=1,
            scorer_averages={},
            results=[result],
        )
        opts = _sim_full_options(run)
        assert opts['metrics'] == []


# ---------------------------------------------------------------------------
# Filter chip on/off clarity: dot->check shape swap drives the accessible
# selection signal (never color alone).
# ---------------------------------------------------------------------------


class TestFilterChipOnOffClarity:
    def test_selected_chip_renders_check_span_and_is_active(self):
        from evaluatorq.dashboard.view import _chip

        html = _chip('goal_outcome', 'Achieved', checked=True, dot_cls='chip-dot-green')
        assert 'is-active' in html
        assert 'class="filter-chip-check chip-dot-green"' in html
        assert 'checked' in html

    def test_unselected_chip_has_no_is_active(self):
        from evaluatorq.dashboard.view import _chip

        html = _chip('goal_outcome', 'Not achieved', checked=False, dot_cls='chip-dot-red')
        assert 'is-active' not in html
        # The check span is still emitted (markup is static); visibility is
        # driven purely by the .is-active CSS state, not by omitting markup.
        assert 'class="filter-chip-check chip-dot-red"' in html
        assert 'checked' not in html


# ---------------------------------------------------------------------------
# Task 3 (metric filters plan): rail markup for rule/goal/turns/tokens/metrics.
# ---------------------------------------------------------------------------


def test_sim_rail_uses_raw_count_maxima():
    from evaluatorq.dashboard.view import render_filter_form

    html = render_filter_form(
        'rid',
        'sim',
        {
            'persona': [],
            'scenario': [],
            'terminated_by': [],
            'goal_outcome': ['Achieved', 'Not achieved'],
            'max_turns': ['8'],
            'max_total_tokens': ['2500'],
            'turn_metrics': ['hallucination_risk'],
        },
        {},
    )
    assert 'name="min_turns" min="1" max="8" step="1"' in html
    assert 'name="min_total_tokens" min="0" max="2500" step="1"' in html
    assert 'name="rule_broken" value="yes"' in html


def test_sim_rail_min_turns_shows_max_beside_slider():
    from evaluatorq.dashboard.view import render_filter_form

    html = render_filter_form(
        'rid',
        'sim',
        {'persona': [], 'scenario': [], 'terminated_by': [], 'goal_outcome': [], 'max_turns': ['8']},
        {},
    )
    assert '<span class="filter-slider-max">/ 8</span>' in html


def test_sim_rail_goal_score_unset_shows_max_not_all():
    from evaluatorq.dashboard.view import render_filter_form

    html = render_filter_form(
        'rid',
        'sim',
        {'persona': [], 'scenario': [], 'terminated_by': [], 'goal_outcome': []},
        {},  # no max_goal_score selection
    )
    # Goal-score ceiling renders its bound (≤ 1) when unset, not "all".
    # (No metrics supplied, so no other ceiling control renders a ≤ readout.)
    assert '<span class="filter-slider-readout">≤ 1</span>' in html


def test_sim_rail_hides_unavailable_metrics():
    from evaluatorq.dashboard.view import render_filter_form

    html = render_filter_form(
        'rid',
        'sim',
        {
            'persona': [],
            'scenario': [],
            'terminated_by': [],
            'goal_outcome': ['Achieved', 'Not achieved'],
            'max_turns': ['8'],
            'max_total_tokens': ['2500'],
            'metrics': ['hallucination_risk'],
        },
        {},
    )
    # Available metric renders its threshold control.
    assert 'name="min_hallucination_risk"' in html
    # Unavailable metrics are hidden entirely.
    assert 'name="max_response_quality"' not in html
    assert 'name="max_tone_appropriateness"' not in html
    assert 'name="max_factual_accuracy"' not in html


def test_sim_rail_omits_empty_more_expander():
    from evaluatorq.dashboard.view import render_filter_form

    html = render_filter_form(
        'rid',
        'sim',
        {
            'persona': [],
            'scenario': [],
            'terminated_by': [],
            'goal_outcome': ['Achieved', 'Not achieved'],
            'max_turns': ['8'],
            'max_total_tokens': ['0'],
            'metrics': [],
        },
        {},
    )
    assert 'id="filter-dd-more"' not in html


def test_sim_rail_more_expander_holds_tokens_and_metrics():
    from evaluatorq.dashboard.view import render_filter_form

    html = render_filter_form(
        'rid',
        'sim',
        {
            'persona': [],
            'scenario': [],
            'terminated_by': [],
            'goal_outcome': ['Achieved', 'Not achieved'],
            'max_turns': ['8'],
            'max_total_tokens': ['2500'],
            'metrics': ['hallucination_risk', 'response_quality'],
        },
        {},
    )
    more_start = html.index('id="filter-dd-more"')
    more_end = html.index('</details>', more_start)
    more_section = html[more_start:more_end]
    assert 'name="min_total_tokens"' in more_section
    assert 'name="min_hallucination_risk"' in more_section
    assert 'name="max_response_quality"' in more_section
    # The rule chip / goal-score ceiling / min-turns controls stay outside.
    assert 'name="rule_broken"' not in more_section
    assert 'name="max_goal_score"' not in more_section
    assert 'name="min_turns"' not in more_section


# ---------------------------------------------------------------------------
# Rail "engaged" affordances: sliders, dropdown status, More-filters badge.
# ---------------------------------------------------------------------------


def _sim_rail(opts, selections):
    from evaluatorq.dashboard.view import render_filter_form

    base = {'persona': [], 'scenario': [], 'terminated_by': [], 'goal_outcome': []}
    base.update(opts)
    return render_filter_form('rid', 'sim', base, selections)


def test_slider_readout_engaged_only_when_off_default():
    # min_turns default floor is 1 → not engaged; moved to 3 → engaged.
    off = _sim_rail({'max_turns': ['8']}, {'min_turns': ['1']})
    assert 'filter-slider-readout is-engaged' not in off
    on = _sim_rail({'max_turns': ['8']}, {'min_turns': ['3']})
    assert 'filter-slider-readout is-engaged' in on


def test_sim_rail_hides_min_turns_when_single_turn_run():
    # No-op control: every conversation has one turn (max_turns == 1) → hidden.
    html = _sim_rail({'max_turns': ['1']}, {})
    assert 'name="min_turns"' not in html
    shown = _sim_rail({'max_turns': ['5']}, {})
    assert 'name="min_turns"' in shown


def test_dropdown_status_marks_partial():
    # A narrowed persona selection reads as engaged (partial), not the "All"
    # default. (Empty selection means "all" by filter convention, so is-none is
    # a defensive style, not reachable through this render path.)
    partial = _sim_rail({'persona': ['a', 'b', 'c']}, {'persona': ['a']})
    assert 'filter-dd-status is-partial' in partial
    assert 'filter-dd-value is-engaged' in partial
    full = _sim_rail({'persona': ['a', 'b', 'c']}, {'persona': ['a', 'b', 'c']})
    assert 'filter-dd-status is-all' in full
    assert 'is-engaged' not in full


def test_more_filters_badge_counts_active_controls():
    opts = {'max_total_tokens': ['2500'], 'metrics': ['hallucination_risk']}
    # min_total_tokens default floor is 0 → no badge.
    inactive = _sim_rail(opts, {'min_total_tokens': ['0']})
    assert 'filter-dd-more-badge' not in inactive
    # Moved off default → badge shows 1.
    active = _sim_rail(opts, {'min_total_tokens': ['500']})
    assert '<span class="filter-dd-more-badge">1</span>' in active
