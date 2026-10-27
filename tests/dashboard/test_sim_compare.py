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
    from evaluatorq.dashboard.sim_compare import build_comparison

    return build_comparison('ra', 'rb', _run_a(), _run_b(), selections or {})


def test_aggregates_per_run():
    from evaluatorq.dashboard.sim_compare import aggregates

    cmp = _cmp()
    a, b = aggregates(cmp.a), aggregates(cmp.b)
    assert (a.conversations, b.conversations) == (3, 3)
    assert a.goal_rate == pytest.approx(2 / 3)
    assert b.goal_rate == pytest.approx(2 / 3)
    assert a.avg_turns == pytest.approx(4.0)
    assert (a.errors, b.errors) == (0, 0)


def test_pair_change_classifies_regressed_and_improved():
    from evaluatorq.dashboard.sim_compare import pair_change

    cmp = _cmp()
    by_persona = {p.a.persona: pair_change(cmp.a, cmp.b, p) for p in cmp.matching.matched}
    # alice: met -> missed; bob: missed -> met
    assert by_persona == {'alice': 'regressed', 'bob': 'improved'}


def test_score_rows_keep_one_sided_scorer_as_none():
    from evaluatorq.dashboard.sim_compare import ScoreRow, score_rows

    from evaluatorq.dashboard.sim_compare import build_comparison

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
    from evaluatorq.dashboard.sim_compare import aggregates

    cmp = _cmp({'goal_outcome': ['Achieved']})
    assert {p.a.persona for p in cmp.matching.matched} == {'alice', 'bob'}
    assert [e.persona for e in cmp.a.entries] == ['alice', 'bob', 'carol']
    assert [e.persona for e in cmp.b.entries] == ['alice', 'bob', 'dave']
    assert aggregates(cmp.a).goal_rate == pytest.approx(2 / 3)
    assert aggregates(cmp.b).goal_rate == pytest.approx(2 / 3)
    assert cmp.total_rows == 4


def test_pair_change_missed_to_error_is_error_not_regressed():
    from evaluatorq.dashboard.sim_compare import build_comparison, pair_change

    run_a = _run('A', [_result('alice', 'billing', goal=False, score=0.1, turns=2)], {})
    run_b = _run('B', [_result('alice', 'billing', goal=False, score=0.0, turns=1, terminated='error')], {})
    cmp = build_comparison('ra', 'rb', run_a, run_b, {})
    assert pair_change(cmp.a, cmp.b, cmp.matching.matched[0]) == 'error'
    # the reverse direction is an error change too, not an improvement
    cmp = build_comparison('rb', 'ra', run_b, run_a, {})
    assert pair_change(cmp.a, cmp.b, cmp.matching.matched[0]) == 'error'


def test_aggregates_count_errored_conversations_in_the_denominator():
    from evaluatorq.dashboard.sim_compare import aggregates, build_comparison
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
    from evaluatorq.dashboard.sim_compare import _scores_panel, build_comparison, score_rows

    run_a, run_b = _run_a(), _run_b()
    for r in run_a.results:
        r.metadata['evaluator_scores'] = {'safety': 0.5}
    for r in run_b.results:
        r.metadata['evaluator_errors'] = {'safety': 'boom'}
    rows = {r.label: r for r in score_rows(build_comparison('ra', 'rb', run_a, run_b, {}))}
    assert rows['safety'].b is None and rows['safety'].dropped_b == 3
    html = _scores_panel(list(rows.values()))
    assert 'failed' in html
    assert 'scorer failed on 3 conversations' in html


def test_scores_panel_draws_tracks_only_within_unit_range():
    from evaluatorq.dashboard.sim_compare import ScoreRow, _scores_panel

    inside = _scores_panel([ScoreRow('ok', 0.2, 0.4)])
    assert 'cmp-track-fill' in inside and 'Mean per conversation, 0 to 1' in inside
    outside = _scores_panel([ScoreRow('big', 3.0, 5.0)])
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
    from evaluatorq.dashboard.sim_compare import match_entries
    from evaluatorq.simulation.reports.sections import individual_entries

    entry = individual_entries(_run_a().results[:1])[0]
    matching = match_entries([entry, entry], [entry, entry])
    assert [(pair.a, pair.b) for pair in matching.matched] == [(entry, entry), (entry, entry)]
    assert matching.a_only == []
    assert matching.b_only == []


def test_match_entries_pairs_and_flags_unmatched():
    from evaluatorq.dashboard.sim_compare import match_entries
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


def test_compare_transcript_route_scopes_session_slug_to_one_response(
    roots: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.dashboard import orq_scope, orq_workspace, sim_compare
    from evaluatorq.dashboard.trace_links import thread_trace_url

    for name in ('ORQ_API_KEY', 'ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(roots[0] / 'settings.json'))
    monkeypatch.setattr(orq_workspace.shutil, 'which', lambda _name: '/usr/bin/orq')
    slugs = iter(('first-workspace', 'second-workspace'))
    calls = 0

    def discover(profile: str | None, *, use_cli_session: bool = False):
        nonlocal calls
        calls += 1
        assert profile is None and use_cli_session is True
        return orq_scope.OrqScope(workspace_key=next(slugs))

    monkeypatch.setattr(orq_scope, 'discover_orq_scope', discover)
    monkeypatch.setattr(
        sim_compare, 'render_transcript_fragment', lambda entry, **_kw: thread_trace_url(str(entry.index)) or ''
    )
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)

    first = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=0&ib=0').text
    assert calls == 1
    assert first.count('/first-workspace/traces?') == 2

    second = client.get(f'/compare/sim/transcript?a={rid_a}&b={rid_b}&ia=0&ib=0').text
    assert calls == 2
    assert second.count('/second-workspace/traces?') == 2


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
    from evaluatorq.dashboard.sim_compare import _headlines, _tiles, score_rows

    cmp = _cmp({'goal_outcome': ['Achieved']})
    assert cmp.outcome_filtered
    heads = _headlines(cmp, score_rows(cmp), 'all')
    assert 'Goal rate not comparable' in heads
    assert 'Goal rate held steady' not in heads and 'Goal rate rose' not in heads
    assert _tiles(cmp).count('not comparable') == 2
    # a persona filter selects on something else, so the rate stays comparable
    assert not _cmp({'persona': ['alice']}).outcome_filtered


def test_biggest_score_move_skips_the_goal_scorer():
    from evaluatorq.dashboard.sim_compare import ScoreRow, _headlines

    cmp = _cmp()
    rows = [ScoreRow('goal_achieved', 1.0, 0.2), ScoreRow('criteria_met', 0.5, 0.4)]
    heads = _headlines(cmp, rows, 'changed')
    assert 'Biggest score move: criteria_met worsened' in heads
    assert 'goal_achieved' not in heads


def test_filter_bar_lists_active_filters_with_remove_links(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&persona=alice&goal_outcome=Achieved').text
    bar = html.split('class="cmp-fbar"')[1].split('</div>')[0]
    assert 'Showing <b>1</b> of <b>4</b> rows' in bar
    assert 'Persona</span>alice' in bar and 'Goal outcome</span>Goal met' in bar
    # removing the persona chip keeps the goal-outcome filter, and clear all drops both
    assert f'href="/compare/sim?a={rid_a}&amp;b={rid_b}&amp;goal_outcome=Achieved"' in bar
    assert f'href="/compare/sim?a={rid_a}&amp;b={rid_b}">Clear all' in bar
    unfiltered = client.get(f'/compare/sim?a={rid_a}&b={rid_b}').text
    assert 'Clear all' not in unfiltered.split('class="cmp-fbar"')[1].split('</div>')[0]


def test_headline_row_toggles_and_follows_the_table(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    page = client.get(f'/compare/sim?a={rid_a}&b={rid_b}&cmp_view=regressed').text
    pressed = re.search(r'<button type="button" class="cmp-head cmp-head--bad is-on" aria-pressed="true" ([^>]*)>', page)
    assert pressed and 'Showing these' in page
    # pressing the showing row again returns the table to its default view
    assert 'cmp_view="' in pressed.group(1)
    conv = client.get(f'/compare/sim/conversations?a={rid_a}&b={rid_b}&cmp_view=improved').text
    oob = conv.split('id="cmp-heads"')[1]
    assert 'hx-swap-oob="true"' in oob.split('>')[0]
    assert re.search(r'class="cmp-head cmp-head--good is-on" aria-pressed="true"', oob)


def test_single_termination_reason_hides_the_ended_bar():
    from evaluatorq.dashboard.sim_compare import _outcomes_panel

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
    assert 'nth-child(1)' in html


def test_first_divergence_and_flipped_criteria():
    from evaluatorq.dashboard.sim_compare import _flipped_criteria, first_divergence
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
    assert _flipped_criteria(a.model_copy(update={'criteria': [row]}), b.model_copy(update={'criteria': [row]})) == []
    assert _flipped_criteria(a.model_copy(update={'criteria': [row]}), b.model_copy(update={'criteria': [flipped]})) == [
        (row.description, row.state, flipped.state)
    ]


def test_rail_goal_outcome_chips_read_goal_met_and_missed(roots: list[Path]):
    client = TestClient(build_app(roots))
    rid_a, rid_b = _rids(roots)
    html = client.get(f'/compare/sim?a={rid_a}&b={rid_b}').text
    rail = html.split('id="filter-form"')[1]
    assert 'value="Achieved"' in rail and 'Goal met' in rail and '>Achieved<' not in rail
