"""Side-by-side comparison of two sim runs.

Covers the aggregate and pair-change math, (persona, scenario) matching incl.
unmatched handling, the compare page, filter, conversation-table and transcript
routes, the graceful-error paths, and the compare picker on the sim overview.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id

# ---------------------------------------------------------------------------
# Fixture builders (real SimulationRun objects — the compare path model_validates)
# ---------------------------------------------------------------------------


def _result(
    persona: str,
    scenario: str,
    *,
    goal: bool,
    score: float,
    turns: int,
    terminated: str = 'judge',
    msgs: list[tuple[str, str]] | None = None,
):
    from evaluatorq.contracts import Message, TokenUsage
    from evaluatorq.simulation.contracts import SimulationResult, TerminatedBy

    return SimulationResult(
        messages=[Message(role=r, content=c) for r, c in (msgs or [])],
        terminated_by=TerminatedBy(terminated),
        reason='done',
        goal_achieved=goal,
        goal_completion_score=score,
        rules_broken=[],
        turn_count=turns,
        turn_metrics=[],
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        metadata={'persona': persona, 'scenario': scenario},
    )


def _run(name: str, results: list, averages: dict[str, float]):
    from evaluatorq.simulation.contracts import SimulationRun

    return SimulationRun(
        run_name=name,
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=len(results),
        scorer_averages=averages,
        results=results,
    )


def _run_a():
    return _run(
        'run-A',
        [
            _result(
                'alice', 'billing', goal=True, score=0.9, turns=4, msgs=[('user', 'hi A'), ('assistant', 'hello A')]
            ),
            _result('bob', 'refund', goal=False, score=0.2, turns=6),
            _result('carol', 'onlyA', goal=True, score=1.0, turns=2),
        ],
        {'goal_achieved': 0.66, 'criteria_met': 0.5},
    )


def _run_b():
    return _run(
        'run-B',
        [
            # alice/billing flips True -> False, score drops
            _result(
                'alice', 'billing', goal=False, score=0.4, turns=5, msgs=[('user', 'hi B'), ('assistant', 'hello B')]
            ),
            _result('bob', 'refund', goal=True, score=0.8, turns=3, terminated='max_turns'),
            _result('dave', 'onlyB', goal=True, score=0.7, turns=2),
        ],
        {'goal_achieved': 0.66, 'safety': 1.0},
    )


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'a.json').write_text(_run_a().model_dump_json())
    (sim / 'b.json').write_text(_run_b().model_dump_json())
    return [tmp_path / 'runs', sim]


def _rids(roots: list[Path]) -> tuple[str, str]:
    sim = roots[1]
    return report_id(sim / 'a.json'), report_id(sim / 'b.json')


# ---------------------------------------------------------------------------
# Unit: aggregates, pair changes, score rows
# ---------------------------------------------------------------------------


def _cmp(selections: dict[str, list[str]] | None = None):
    from evaluatorq.dashboard.sim_compare.model import build_comparison

    return build_comparison('ra', 'rb', _run_a(), _run_b(), selections or {})


def test_aggregates_per_run():
    from evaluatorq.dashboard.sim_compare.model import aggregates

    cmp = _cmp()
    a, b = aggregates(cmp.a), aggregates(cmp.b)
    assert (a.conversations, b.conversations) == (3, 3)
    assert a.goal_rate == pytest.approx(2 / 3)
    assert b.goal_rate == pytest.approx(2 / 3)
    assert a.avg_turns == pytest.approx(4.0)
    assert (a.errors, b.errors) == (0, 0)


def test_pair_change_classifies_regressed_and_improved():
    from evaluatorq.dashboard.sim_compare.model import pair_change

    cmp = _cmp()
    by_persona = {p.a.persona: pair_change(cmp.a, cmp.b, p) for p in cmp.matching.matched}
    # alice: met -> missed; bob: missed -> met
    assert by_persona == {'alice': 'regressed', 'bob': 'improved'}


def test_score_rows_keep_one_sided_scorer_as_none():
    from evaluatorq.dashboard.sim_compare.model import ScoreRow, score_rows

    from evaluatorq.dashboard.sim_compare.model import build_comparison

    run_a, run_b = _run_a(), _run_b()
    run_a.results[0].metadata['evaluator_scores'] = {'shared': 0.5, 'only_a': 1.0}
    run_b.results[0].metadata['evaluator_scores'] = {'shared': 0.75, 'only_b': 0.25}

    rows = {r.label: r for r in score_rows(build_comparison('ra', 'rb', run_a, run_b, {}))}
    assert rows['shared'].delta == pytest.approx(0.25)
    # A scorer only one run measured stays None (n/a), never a fabricated 0.0 regression.
    assert rows['only_a'] == ScoreRow('only_a', 1.0, None)
    assert rows['only_b'].a is None
    assert rows['only_b'].delta is None


def test_filter_keeps_pair_when_either_side_matches():
    # goal_outcome=Achieved: A's alice and carol and B's bob and dave match. Both pairs survive
    # (alice, bob) plus both unmatched entries, and each side is described by its half of those rows.
    from evaluatorq.dashboard.sim_compare.model import aggregates

    cmp = _cmp({'goal_outcome': ['Achieved']})
    assert {p.a.persona for p in cmp.matching.matched} == {'alice', 'bob'}
    assert [e.persona for e in cmp.a.entries] == ['alice', 'bob', 'carol']
    assert [e.persona for e in cmp.b.entries] == ['alice', 'bob', 'dave']
    assert aggregates(cmp.a).goal_rate == pytest.approx(2 / 3)
    assert aggregates(cmp.b).goal_rate == pytest.approx(2 / 3)
    assert cmp.total_rows == 4


def test_pair_change_missed_to_error_is_error_not_regressed():
    from evaluatorq.dashboard.sim_compare.model import build_comparison, pair_change

    run_a = _run('A', [_result('alice', 'billing', goal=False, score=0.1, turns=2)], {})
    run_b = _run('B', [_result('alice', 'billing', goal=False, score=0.0, turns=1, terminated='error')], {})
    cmp = build_comparison('ra', 'rb', run_a, run_b, {})
    assert pair_change(cmp.a, cmp.b, cmp.matching.matched[0]) == 'error'
    # the reverse direction is an error change too, not an improvement
    cmp = build_comparison('rb', 'ra', run_b, run_a, {})
    assert pair_change(cmp.a, cmp.b, cmp.matching.matched[0]) == 'error'


def test_aggregates_count_errored_conversations_in_the_denominator():
    from evaluatorq.dashboard.sim_compare.model import aggregates, build_comparison
    from evaluatorq.simulation.reports.sections import _build_summary_section

    run_a = _run(
        'A',
        [
            _result('alice', 'billing', goal=True, score=1.0, turns=2),
            _result('bob', 'refund', goal=False, score=0.0, turns=1, terminated='error'),
        ],
        {},
    )
    cmp = build_comparison('ra', 'rb', run_a, _run_b(), {})
    agg = aggregates(cmp.a)
    assert agg.goal_rate == pytest.approx(0.5)
    assert agg.errors == 1
    summary = _build_summary_section(run_a.results).data
    assert agg.goal_rate == summary['success_rate']
    assert agg.avg_score == summary['avg_goal_completion_score']


def test_scorer_with_every_result_dropped_renders_as_failed():
    from evaluatorq.dashboard.sim_compare.panels import _scores_panel
    from evaluatorq.dashboard.sim_compare.model import build_comparison, score_rows

    run_a, run_b = _run_a(), _run_b()
    for r in run_a.results:
        r.metadata['evaluator_scores'] = {'safety': 0.5}
    for r in run_b.results:
        r.metadata['evaluator_errors'] = {'safety': 'boom'}
    rows = {r.label: r for r in score_rows(build_comparison('ra', 'rb', run_a, run_b, {}))}
    assert rows['safety'].b is None and rows['safety'].dropped_b == 3
    html = _scores_panel(list(rows.values())) or ''
    assert 'failed' in html
    assert 'scorer failed on 3 conversations' in html


def test_scores_panel_draws_tracks_only_within_unit_range():
    from evaluatorq.dashboard.sim_compare.model import ScoreRow
    from evaluatorq.dashboard.sim_compare.panels import _scores_panel

    inside = _scores_panel([ScoreRow('ok', 0.2, 0.4)]) or ''
    assert 'cmp-track-fill' in inside and 'Mean per conversation, 0 to 1' in inside
    outside = _scores_panel([ScoreRow('big', 3.0, 5.0)]) or ''
    assert 'cmp-track-fill' not in outside
    assert '0 to 1' not in outside


def test_default_sort_puts_regressed_first(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim/conversations?a={rid_a}&b={rid_b}&cmp_view=all').text
    assert html.index('sim-conv-persona">alice<') < html.index('sim-conv-persona">bob<')


# ---------------------------------------------------------------------------
# Unit: matching by (persona, scenario)
# ---------------------------------------------------------------------------


def test_match_entries_counts_repeated_same_object_by_position() -> None:
    from evaluatorq.dashboard.sim_compare.model import match_entries
    from evaluatorq.simulation.reports.sections import individual_entries

    entry = individual_entries(_run_a().results[:1])[0]
    matching = match_entries([entry, entry], [entry, entry])
    assert [(pair.a, pair.b) for pair in matching.matched] == [(entry, entry), (entry, entry)]
    assert matching.a_only == []
    assert matching.b_only == []


def test_match_entries_pairs_and_flags_unmatched():
    from evaluatorq.dashboard.sim_compare.model import match_entries
    from evaluatorq.simulation.reports.sections import individual_entries

    m = match_entries(individual_entries(_run_a().results), individual_entries(_run_b().results))
    matched_keys = {(p.a.persona, p.a.scenario) for p in m.matched}
    assert matched_keys == {('alice', 'billing'), ('bob', 'refund')}
    assert [(e.persona, e.scenario) for e in m.a_only] == [('carol', 'onlyA')]
    assert [(e.persona, e.scenario) for e in m.b_only] == [('dave', 'onlyB')]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_compare_page_renders_headlines_tiles_and_table(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}&b={rid_b}')
    assert resp.status_code == 200
    html = resp.text
    assert 'run-A' in html and 'run-B' in html
    # "what changed" headlines: one regression, one improvement
    assert 'Regressed in B' in html and 'Improved in B' in html
    assert 'Goal rate held steady' in html
    # tiles, panels, filter rail, drawer shell
    for label in ('Conversations', 'Goal rate', 'Avg goal score', 'Avg turns', 'Errors'):
        assert f'<div class="kpi-label">{label}</div>' in html
    assert 'How conversations ended' in html
    assert 'id="filter-form"' in html
    assert 'sim-entity-dialog--wide' in html
    # segment control defaults to changed conversations, with counts
    assert re.search(r'aria-pressed="true">Changed<span class="cmp-seg-n">2</span>', html)
    # rows open the paired transcript in the drawer
    assert 'data-drawer-url="/compare/sim/transcript?' in html
    # unmatched conversations surfaced, not dropped
    assert 'Only in A' in html and 'Only in B' in html
    assert 'carol' in html and 'dave' in html


def test_compare_filter_route_narrows_table_and_tiles(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    resp = client.post(
        f'/compare/sim/filter?a={rid_a}&b={rid_b}',
        data={'persona': 'alice', 'cmp_view': 'all'},
    )
    assert resp.status_code == 200
    html = resp.text
    assert 'id="filter-swap"' in html
    assert 'sim-conv-persona">alice<' in html
    assert 'sim-conv-persona">bob<' not in html
    # table state from the hidden inputs survives the filter post
    assert re.search(r'aria-pressed="true">All<span class="cmp-seg-n">1</span>', html)


def test_compare_conversations_route_switches_view(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    resp = client.get(f'/compare/sim/conversations?a={rid_a}&b={rid_b}&cmp_view=improved')
    assert resp.status_code == 200
    html = resp.text
    assert 'id="cmp-conversations"' in html
    assert 'sim-conv-persona">bob<' in html
    assert 'sim-conv-persona">alice<' not in html


def test_compare_conversations_route_clamps_bad_params(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    resp = client.get(
        f'/compare/sim/conversations?a={rid_a}&b={rid_b}&cmp_view=nope&cmp_sort=x&cmp_page=-3&cmp_size=7'
    )
    assert resp.status_code == 200
    html = resp.text
    # bad view, sort, page and size fall back to the defaults
    assert 'aria-pressed="true">Changed' in html
    assert re.search(r'<th scope="col" aria-sort="ascending"><button[^>]*>Change', html)
    assert 'cmp_page' not in html.split('id="cmp-table-state"')[1].split('</div>')[0]
    assert 'name="cmp_size" value="10"' in html


def test_filter_and_conversations_routes_push_a_canonical_url(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    filtered = client.post(
        f'/compare/sim/filter?a={rid_a}&b={rid_b}', data={'persona': 'alice', 'cmp_view': 'all', 'cmp_size': '25'}
    )
    assert filtered.headers['HX-Push-Url'] == f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice&cmp_view=all&cmp_size=25'
    conv = client.get(f'/compare/sim/conversations?a={rid_a}&b={rid_b}&persona=alice&cmp_view=improved&cmp_page=2')
    assert conv.headers['HX-Push-Url'] == (
        f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice&cmp_view=improved&cmp_page=2'
    )


def test_get_route_reproduces_the_pushed_view(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice&cmp_view=all').text
    assert re.search(r'aria-pressed="true">All<span class="cmp-seg-n">1</span>', html)
    assert 'sim-conv-persona">alice<' in html
    assert 'sim-conv-persona">bob<' not in html


def test_compare_transcript_route_shows_both_sides(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    # alice/billing is index 0 in both runs
    resp = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=0&ib=0')
    assert resp.status_code == 200
    html = resp.text
    assert 'hello A' in html and 'hello B' in html
    assert 'cmp-transcript-grid' in html


def test_compare_transcript_rejects_a_non_matched_pair(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    # A's alice (0) with B's bob (1) are both real conversations, but not a pair
    resp = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=0&ib=1')
    assert resp.status_code == 404
    assert 'Not a matched pair.' in resp.text


def test_compare_missing_param_is_400(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, _ = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}')
    assert resp.status_code == 400


def test_compare_unknown_rid_is_404(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, _ = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}&b=deadbeef')
    assert resp.status_code == 404


def test_compare_self_compare_is_400(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, _ = _rids(roots)
    resp = client.get(f'/compare/sim?a={rid_a}&b={rid_a}')
    assert resp.status_code == 400


def test_compare_empty_runs_render_na_tiles_without_failure(tmp_path: Path):
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'a.json').write_text(_run('empty-A', [], {}).model_dump_json())
    (sim / 'b.json').write_text(_run('empty-B', [], {}).model_dump_json())
    rid_a = report_id(sim / 'a.json')
    rid_b = report_id(sim / 'b.json')

    resp = TestClient(build_app([tmp_path / 'runs', sim])).get(f'/compare/sim?a={rid_a}&b={rid_b}')
    assert resp.status_code == 200
    html = resp.text
    assert '<div class="kpi-value">n/a</div>' in html
    assert 'No conversations' in html
    assert 'No conversations matched' in html


def test_compare_partially_empty_run_keeps_outcomes_panel(tmp_path: Path):
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'a.json').write_text(_run('empty-A', [], {}).model_dump_json())
    (sim / 'b.json').write_text(
        _run('nonempty-B', [_result('alice', 'billing', goal=True, score=1.0, turns=1)], {}).model_dump_json()
    )
    rid_a = report_id(sim / 'a.json')
    rid_b = report_id(sim / 'b.json')

    resp = TestClient(build_app([tmp_path / 'runs', sim])).get(f'/compare/sim?a={rid_a}&b={rid_b}')
    assert resp.status_code == 200
    assert 'Goal outcome' in resp.text or 'goal outcome' in resp.text.lower()
    assert 'Only in B' in resp.text


def test_compare_corrupt_report_is_422(tmp_path: Path):
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'good.json').write_text(_run_a().model_dump_json())
    # a file that sniffs as sim ('mode' present) but fails model validation
    (sim / 'bad.json').write_text('{"mode": "run", "results": "not-a-list"}')
    from evaluatorq.dashboard.library import report_id

    client = TestClient(build_app([tmp_path / 'runs', sim]))
    good = report_id(sim / 'good.json')
    bad = report_id(sim / 'bad.json')
    resp = client.get(f'/compare/sim?a={good}&b={bad}')
    assert resp.status_code == 422


def test_compare_syntactically_corrupt_report_is_422(tmp_path: Path):
    # A file that isn't even valid JSON must be reported as corrupt (422), not masked
    # as "not found" (404). Guards the load_surface_strict path in resolve_run.
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'good.json').write_text(_run_a().model_dump_json())
    (sim / 'bad.json').write_text('{not valid json at all')
    from evaluatorq.dashboard.library import report_id

    client = TestClient(build_app([tmp_path / 'runs', sim]))
    good = report_id(sim / 'good.json')
    bad = report_id(sim / 'bad.json')
    resp = client.get(f'/compare/sim?a={good}&b={bad}')
    assert resp.status_code == 422


def test_compare_transcript_invalid_index_is_400(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    resp = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=x&ib=0')
    assert resp.status_code == 400


def test_overview_drops_compare_selectors(roots: list[Path]):
    client = TestClient(build_app(roots))
    html = client.get('/?surface=sim').text
    assert 'Compare runs' not in html
    assert 'name="a"' not in html and 'name="b"' not in html


def test_compare_bar_absent_on_sim_overview(roots: list[Path]):
    client = TestClient(build_app(roots))
    resp = client.get('/?surface=sim')
    assert resp.status_code == 200
    html = resp.text
    assert 'action="/compare/sim"' not in html
    assert 'Recent runs' in html


def test_compare_control_on_sim_report(roots: list[Path]):
    """The run report hero carries a compare control: this run pinned as A
    (hidden input), the other run selectable as B — never itself."""
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/r/{rid_a}').text
    assert f'name="a" value="{rid_a}"' in html
    assert f'<option value="{rid_b}"' in html
    assert f'<option value="{rid_a}"' not in html


# ---------------------------------------------------------------------------
# Critique follow-ups: honest goal rate, filter bar, headline toggle, drawer diff
# ---------------------------------------------------------------------------


def test_outcome_filter_makes_goal_rate_not_comparable():
    from evaluatorq.dashboard.sim_compare.heatmap import where_panel
    from evaluatorq.dashboard.sim_compare.model import ScoreRow
    from evaluatorq.dashboard.sim_compare.panels import render_headlines, _scores_panel, _tiles

    cmp = _cmp({'goal_outcome': ['Achieved']})
    assert cmp.outcome_filter == 'goal outcome'
    heads = render_headlines(cmp, 'all')
    assert 'Goal rate hidden' in heads
    assert 'Goal rate held steady' not in heads and 'Goal rate rose' not in heads
    tiles = _tiles(cmp)
    assert tiles.count('Hidden: you filtered by goal outcome') == 2
    # the hidden tiles drop their A -> B values; the other three keep them
    assert tiles.count('cmp-tile-ab') == 3
    # the goal scorer row and the heatmap's goal-rate change hide too
    rows = [ScoreRow('goal_achieved', 0.5, 0.75), ScoreRow('helpfulness', 0.6, 0.7)]
    scores = _scores_panel(rows, cmp.outcome_filter) or ''
    assert 'Hidden: goal_achieved, because you filtered by goal outcome' in scores
    assert 'cmp-score-label">goal_achieved' not in scores and 'cmp-score-label">helpfulness' in scores
    assert 'Hidden: goal_achieved' not in (_scores_panel(rows) or '')
    assert ' pts<' not in where_panel(cmp)
    assert 'pair' in where_panel(cmp)
    assert _cmp({'max_goal_score': ['0.5']}).outcome_filter == 'goal score'
    # a persona filter selects on something else, so the rate stays comparable
    assert not _cmp({'persona': ['alice']}).outcome_filter


def test_headline_count_unit_and_flat_tiles_stay_quiet():
    from evaluatorq.dashboard.sim_compare.panels import _outcomes_and_scores, _tiles, render_headlines
    from evaluatorq.dashboard.sim_compare.state import TableState

    cmp = _cmp()
    heads = render_headlines(cmp, 'all')
    # the count is the big number; the unit word is set small beside it
    assert '<span class="rk-head-number">1 <span class="rk-head-unit">conversation</span></span>' in heads
    assert '<span class="rk-head-unit">pts</span>' in heads
    # an unchanged tile reads "x → x" without a "no change" pill on top
    assert 'no change' not in _tiles(cmp)
    # no scorer rows: Outcomes takes the row and a note replaces the empty Scores card
    body = _outcomes_and_scores(cmp, TableState())
    assert 'Scores' not in body.split('cmp-scores-note')[0]
    assert 'cmp-scores-note">Neither run recorded scorer' in body


def test_biggest_score_move_skips_the_goal_scorer():
    from evaluatorq.dashboard.sim_compare.model import ScoreRow
    from evaluatorq.dashboard.sim_compare.panels import render_headlines

    cmp = _cmp()
    cmp.__dict__['scores'] = [
        ScoreRow('goal_achieved', 1.0, 0.2),
        ScoreRow('criteria_met', 0.5, 0.4),
        ScoreRow('Turns', 4.0, 9.0, higher_better=False),
    ]
    heads = render_headlines(cmp, 'changed')
    assert 'Biggest score move: criteria_met worsened' in heads
    assert 'goal_achieved' not in heads and 'Turns' not in heads


def test_filter_bar_lists_active_filters_with_remove_links(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice&goal_outcome=Achieved').text
    bar = html.split('class="rk-fbar"')[1].split('</div>')[0]
    assert 'Showing <b>1</b> of <b>4</b> rows' in bar
    assert 'Persona</span>alice' in bar and 'Goal outcome</span>Goal met' in bar
    # removing the persona chip keeps the goal-outcome filter, and clear all drops both
    assert f'href="/compare/sim?a={rid_a}&amp;b={rid_b}&amp;goal_outcome=Achieved"' in bar
    assert f'hx-get="/compare/sim/filter?a={rid_a}&amp;b={rid_b}&amp;goal_outcome=Achieved"' in bar
    assert f'href="/compare/sim?a={rid_a}&amp;b={rid_b}" hx-get="/compare/sim/filter?a={rid_a}&amp;b={rid_b}"' in bar
    assert 'aria-label="Remove the Persona filter"' in bar
    unfiltered = client.get(f'/compare/sim?a={rid_a}&b={rid_b}').text
    assert 'Clear all' not in unfiltered.split('class="rk-fbar"')[1].split('</div>')[0]


def test_headline_row_toggles_and_follows_the_table(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    page = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&cmp_view=regressed').text
    pressed = re.search(r'<div class="rk-head rk-head--click rk-head--bad is-on">.*?<button [^>]*aria-pressed="true" ([^>]*)>', page)
    assert pressed and 'Showing these' in page
    # pressing the showing row again returns the table to its default view
    assert 'cmp_view="' in pressed.group(1)
    conv = client.get(f'/compare/sim/conversations?a={rid_a}&b={rid_b}&cmp_view=improved').text
    oob = conv.split('id="cmp-heads"')[1]
    assert 'hx-swap-oob="true"' in oob.split('>')[0]
    assert re.search(r'class="rk-head rk-head--click rk-head--good is-on">.*?aria-pressed="true"', oob)


def test_single_termination_reason_hides_the_ended_bar():
    from evaluatorq.dashboard.sim_compare.panels import _outcomes_panel

    assert 'How conversations ended' in _outcomes_panel(_cmp())
    assert 'How conversations ended' not in _outcomes_panel(_cmp({'persona': ['alice']}))


def test_drawer_names_the_pair_once_and_lists_differences(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=0&ib=0').text
    assert html.count('alice') == 1
    assert 'sim-conv-summary' not in html
    assert 'aria-label="Differences"' in html
    assert 'First differs at message 1' in html
    assert html.count('sim-msg--marked') == 2


def test_message_list_marks_only_the_given_position():
    from evaluatorq.common.reports.messages import render_message_list

    msgs = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
    html = render_message_list(msgs, role_labels={}, class_prefix='sim', mark_index=1)
    assert html.count('sim-msg--marked') == 1
    assert html.index('sim-msg--marked') > html.index('hi')
    assert 'sim-msg--marked' not in render_message_list(msgs, role_labels={}, class_prefix='sim')


def test_filter_chip_links_follow_the_table_and_swap_through_the_filter_route(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    conv = client.get(f'/compare/sim/conversations?a={rid_a}&b={rid_b}&persona=alice&cmp_view=all').text
    bar = conv.split('id="cmp-fbar"')[1].split('</div>')[0]
    assert 'hx-swap-oob="true"' in bar.split('>')[0]
    assert f'hx-get="/compare/sim/filter?a={rid_a}&amp;b={rid_b}&amp;cmp_view=all"' in bar
    resp = client.get(f'/compare/sim/filter?a={rid_a}&b={rid_b}&cmp_view=all')
    assert resp.status_code == 200 and 'id="filter-swap"' in resp.text
    assert resp.headers['HX-Push-Url'] == f'/compare/sim?a={rid_a}&b={rid_b}&cmp_view=all'


def test_copy_link_reports_failure_instead_of_claiming_success():
    js = (Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js').read_text()
    handler = js.split("closest('[data-copy-link]')")[1].split('});')[0]
    assert '.then(' in handler and 'Copy failed' in handler


def test_compare_params_split_filters_from_table_state_and_drop_unknown_keys():
    from evaluatorq.dashboard.sim_compare.state import parse_compare_params

    selections, state = parse_compare_params(
        [('a', 'x'), ('persona', 'alice'), ('persona', 'bob'), ('cmp_view', 'all'), ('bogus', '1')]
    )
    assert selections == {'persona': ['alice', 'bob']}
    assert state.view == 'all'


def test_first_divergence_and_flipped_criteria():
    from evaluatorq.common.reports.sim_detail import flipped_criteria, first_divergence
    from evaluatorq.dashboard.sim_views import _entries_from_run

    a = _entries_from_run(_run_a())[0]
    b = _entries_from_run(_run_b())[0]
    assert first_divergence(a, a) is None
    assert first_divergence(a, b) == 0
    longer = a.model_copy(update={'transcript': [*a.transcript, a.transcript[0]]})
    assert first_divergence(a, longer) == len(a.transcript)

    row = a.criteria[0] if a.criteria else None
    if row is None:
        from evaluatorq.simulation.contracts import CriteriaRow

        row = CriteriaRow(id='c0', description='asks for order id', type='must_happen', passed=True, safety=False)
    flipped = row.model_copy(update={'passed': not row.passed})
    assert flipped_criteria(a.model_copy(update={'criteria': [row]}), b.model_copy(update={'criteria': [row]})) == []
    assert flipped_criteria(a.model_copy(update={'criteria': [row]}), b.model_copy(update={'criteria': [flipped]})) == [
        (row.description, row.state, flipped.state)
    ]


def test_rail_goal_outcome_chips_read_goal_met_and_missed(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}').text
    rail = html.split('id="filter-form"')[1]
    assert 'value="Achieved"' in rail and 'Goal met' in rail and '>Achieved<' not in rail


def test_comparison_is_cached_per_file_stamp_and_narrowing(roots: list[Path]):
    import os

    from evaluatorq.dashboard import library
    from evaluatorq.dashboard.sim_compare.model import cached_comparison
    from evaluatorq.dashboard.sim_views import _entries_from_run

    paths = (roots[1] / 'a.json', roots[1] / 'b.json')
    stamps = (library.stamp(paths[0]), library.stamp(paths[1]))
    first = cached_comparison(stamps, 'ra', 'rb', {})
    assert cached_comparison(stamps, 'ra', 'rb', {}) is first

    # A selection that drops nothing (the rail posts every control) is the same slice.
    everyone = sorted({e.persona for run in (_run_a(), _run_b()) for e in _entries_from_run(run)})
    assert cached_comparison(stamps, 'ra', 'rb', {'persona': everyone}) is first

    # A selection that narrows is a different slice.
    assert cached_comparison(stamps, 'ra', 'rb', {'goal_outcome': ['Achieved']}) is not first

    # A rewrite changes the stamp, so the same request rebuilds.
    stat = paths[0].stat()
    os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert cached_comparison((library.stamp(paths[0]), stamps[1]), 'ra', 'rb', {}) is not first


def test_comparison_pills_share_the_report_kit_tones():
    from evaluatorq.dashboard.report_kit import pill
    from evaluatorq.dashboard.sim_compare.fmt import status_badge, change_pill, delta_pill

    assert change_pill('regressed') == pill('Regressed', tone='danger', dot=True)
    assert change_pill('improved') == pill('Improved', tone='good', dot=True)
    assert status_badge('error') == pill('Error', tone='warn')
    assert delta_pill(0.2, '0.20') == pill('▲ 0.20', tone='good')
    assert delta_pill(0.2, '0.20', higher_better=False) == pill('▲ 0.20', tone='danger')


def _all_links(html: str, css: str) -> list[str]:
    return re.findall(rf'<a class="{css}[^"]*"[^>]*>', html)


def test_outcome_legend_entry_links_to_its_filter_value():
    from evaluatorq.dashboard.sim_compare.panels import _outcomes_panel

    panel = _outcomes_panel(_cmp())
    met = next(a for a in _all_links(panel, 'cmp-legend-link') if 'Goal met' in a)
    assert 'goal_outcome=Achieved' in met and 'hx-get="/compare/sim/filter?' in met
    assert 'Select to filter' in met and 'is-on' not in met
    # the bar segment carries the same toggle; the error category has no rail value, so no link
    assert any('goal_outcome=Achieved' in a for a in _all_links(panel, 'cmp-seg-link'))
    assert not any('Error' in a for a in _all_links(panel, 'cmp-legend-link'))


def test_clicking_an_active_outcome_value_removes_it():
    from evaluatorq.dashboard.sim_compare.panels import _outcomes_panel

    cmp = _cmp({'goal_outcome': ['Achieved']})
    met = next(a for a in _all_links(_outcomes_panel(cmp), 'cmp-legend-link') if 'Goal met' in a)
    assert 'is-on' in met and 'aria-current="true"' in met and 'Filter on' in met
    assert 'goal_outcome' not in met
    # an inactive value is added beside the active one
    missed = next(a for a in _all_links(_outcomes_panel(cmp), 'cmp-legend-link') if 'Goal missed' in a)
    assert 'goal_outcome=Achieved' in missed and 'goal_outcome=Not+achieved' in missed


def test_termination_reason_toggles_terminated_by():
    from evaluatorq.dashboard.sim_compare.panels import _outcomes_panel

    cmp = _cmp()
    assert len({e.terminated_by for s in (cmp.a, cmp.b) for e in s.entries}) > 1
    assert any('terminated_by=' in a for a in _all_links(_outcomes_panel(cmp), 'cmp-legend-link'))


def test_headline_scenario_is_a_link_outside_the_button():
    from evaluatorq.dashboard.sim_compare.model import build_comparison
    from evaluatorq.dashboard.sim_compare.panels import render_headlines

    def run(name: str, goal: bool):
        return _run(name, [_result(p, 'billing', goal=goal, score=0.5, turns=3) for p in ('ann', 'ben')], {})

    cmp = build_comparison('ra', 'rb', run('A', True), run('B', False), {})
    heads = render_headlines(cmp, 'changed')
    link = re.search(r'<a class="cmp-head-scn" ([^>]*)>billing</a>', heads)
    assert link and 'scenario=billing' in link.group(1) and 'hx-get="/compare/sim/filter?' in link.group(1)
    assert 'most in' in heads and '(2 of 2)' in heads
    for button in re.findall(r'<button.*?</button>', heads, flags=re.S):
        assert '<a ' not in button


def test_empty_state_offers_clear_all_only_with_active_filters(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice&cmp_view=error').text
    empty = html.split('class="cmp-empty"')[1].split('</section>')[0]
    assert 'Clear all filters' in empty and 'Show all conversations' in empty
    # no filters: the empty segment still offers "Show all", but nothing to clear
    bare = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&cmp_view=error').text
    empty = bare.split('class="cmp-empty"')[1].split('</section>')[0]
    assert 'Clear all filters' not in empty and 'Show all conversations' in empty


def test_pairing_rule_shows_only_with_active_filters(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    rule = "A pair stays when either run's conversation matches."
    on = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice').text
    off = client.get(f'/compare/sim?a={rid_a}&b={rid_b}').text
    assert on.count(rule.replace("'", '&#x27;')) + on.count(rule) == 1
    assert rule not in off and rule.replace("'", '&#x27;') not in off


# ---------------------------------------------------------------------------
# "Where it changed" persona x scenario heatmap
# ---------------------------------------------------------------------------


def _where(selections: dict[str, list[str]] | None = None) -> str:
    from evaluatorq.dashboard.sim_compare.heatmap import where_panel

    return where_panel(_cmp(selections))


def _heat_cell(html: str, persona: str, scenario: str) -> str:
    """The ``<td>`` of the heatmap cell for (persona, scenario)."""
    cols = re.findall(r'<th class="rk-heat-col">([^<]*)</th>', html)
    row = re.search(rf'<th class="rk-heat-row">{persona}</th>(.*?)</tr>', html, re.S)
    assert row, persona
    tds = re.findall(r'<td.*?</td>', row.group(1), re.S)
    return tds[cols.index(scenario)]


def test_where_cell_sign_and_tone():
    html = _where()
    worse = _heat_cell(html, 'alice', 'billing')
    better = _heat_cell(html, 'bob', 'refund')
    assert '\N{MINUS SIGN}100 pts' in worse
    assert '+100 pts' in better
    assert 'background:#e58f89' in worse  # red: met in A, missed in B
    assert 'background:#7fc8a5' in better  # green: missed in A, met in B
    assert '1 pair: 1 regressed, 0 improved' in worse
    assert 'rk-heat-empty' in _heat_cell(html, 'alice', 'refund')


def test_where_cell_is_neutral_when_nothing_moved():
    from evaluatorq.dashboard.sim_compare.heatmap import _cell_value
    from evaluatorq.dashboard.sim_compare.model import MatchedPair

    cmp = _cmp()
    by = {c: p for p, c in cmp.changes}
    text, fill, title = _cell_value(cmp, [(by['regressed'], 'regressed'), (by['improved'], 'improved')])
    assert text == '0 pts'
    assert fill == '#eceff1'
    assert '2 pairs: 1 regressed, 1 improved' in title
    assert isinstance(by['regressed'], MatchedPair)


def test_where_cell_counts_errors_apart_from_misses():
    from evaluatorq.dashboard.sim_compare.heatmap import _cell_value

    cmp = _cmp()
    pair = cmp.matching.matched[0]
    errored = cmp.a.__class__(**{**cmp.a.__dict__, 'errored': frozenset({pair.a.index})})
    cmp = cmp.__class__(**{**cmp.__dict__, 'a': errored})
    text, _, title = _cell_value(cmp, [(pair, 'error')])
    assert text == 'err'
    assert '1 with an error' in title


def test_where_cell_link_sets_persona_and_scenario_and_keeps_other_filters():
    html = _where({'terminated_by': ['judge'], 'persona': ['alice']})
    link = re.search(r'href="([^"]*)"', _heat_cell(html, 'alice', 'billing'))
    assert link
    query = link.group(1).replace('&amp;', '&')
    assert 'persona=alice' in query and 'scenario=billing' in query
    assert query.count('persona=') == 1
    assert 'terminated_by=judge' in query


def test_where_active_cell_toggles_off_and_keeps_other_filters():
    html = _where({'persona': ['alice'], 'scenario': ['billing'], 'terminated_by': ['judge']})
    cell = _heat_cell(html, 'alice', 'billing')
    assert 'is-on' in cell and 'aria-current="true"' in cell
    query = re.search(r'href="([^"]*)"', cell).group(1).replace('&amp;', '&')
    assert 'persona=' not in query and 'scenario=' not in query
    assert 'terminated_by=judge' in query


def test_where_panel_hidden_for_one_persona_and_one_scenario():
    from evaluatorq.dashboard.sim_compare.heatmap import where_panel
    from evaluatorq.dashboard.sim_compare.model import build_comparison

    run_a = _run('a', [_result('alice', 'billing', goal=True, score=1, turns=2)], {})
    run_b = _run('b', [_result('alice', 'billing', goal=False, score=0, turns=2)], {})
    assert where_panel(build_comparison('ra', 'rb', run_a, run_b, {})) == ''
    assert 'Where it changed' in _where()


def test_where_panel_caps_columns_and_says_how_many_are_left():
    from evaluatorq.dashboard.sim_compare.heatmap import where_panel
    from evaluatorq.dashboard.sim_compare.model import build_comparison

    names = [f's{i:02d}' for i in range(15)]
    run_a = _run('a', [_result('alice', n, goal=True, score=1, turns=2) for n in names], {})
    run_b = _run('b', [_result('alice', n, goal=False, score=0, turns=2) for n in names], {})
    html = where_panel(build_comparison('ra', 'rb', run_a, run_b, {}))
    assert html.count('rk-heat-col') == 12
    assert '+3 more scenarios' in html


def test_compare_page_places_the_heatmap_after_outcomes(roots: list[Path]):
    rid_a, rid_b = _rids(roots)
    client = TestClient(build_app(roots))
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}').text
    assert html.index('Outcomes') < html.index('Where it changed') < html.index('id="cmp-conversations"')
